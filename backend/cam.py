from flask import Flask, send_file, jsonify
import cv2
import os
import numpy as np
import io
from ultralytics import YOLO
from dotenv import load_dotenv

# Setup Flask dan model YOLO
app = Flask(__name__)
model = YOLO("yolov8.pt")  # Ganti model sesuai kebutuhan

# Load env
load_dotenv()
camera_ip = os.getenv("CAMERA_IP")
camera = camera_ip  # Bisa juga langsung string RTSP/HTTP stream

@app.route('/detect-camera', methods=['GET'])
def detect_from_camera():
    cap = cv2.VideoCapture(camera)

    if not cap.isOpened():
        return jsonify({'error': 'Gagal membuka kamera'}), 500

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': 'Gagal membaca frame dari kamera'}), 500

    # Deteksi objek
    results = model(frame)
    annotated_frame = results[0].plot()

    # Encode frame hasil deteksi
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    img_bytes = io.BytesIO(buffer)

    return send_file(img_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)

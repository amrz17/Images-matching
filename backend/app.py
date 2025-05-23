from flask import Flask, jsonify, Response, request, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from datetime import datetime
from ultralytics import YOLO
import os
import cv2
import easyocr
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from datetime import datetime
import pytz
import json
from dotenv import load_dotenv
import tempfile
import requests
import time
import threading
import subprocess  # Tambahkan ini untuk menjalankan file python lain



load_dotenv()  # Ini akan membaca file .env

# Contoh ambil variabel
database_url = os.getenv("SQLALCHEMY_DATABASE_URI")

# Inisialisasi SQLAlchemy dan Marshmallow
db = SQLAlchemy()
ma = Marshmallow()

def current_time_wib():
    return datetime.now(pytz.timezone("Asia/Jakarta"))

# MODELS
class Vehicle(db.Model):
    __tablename__ = 'vehicles'
    
    id = db.Column(db.Integer, primary_key=True)
    license_plate = db.Column(db.String(20), nullable=False)
    vehicle_type = db.Column(db.String(50), nullable=True)
    entry_time = db.Column(db.DateTime, nullable=False, default=current_time_wib)
    entry_image_path = db.Column(db.String(255), nullable=False)
    feature = db.Column(db.Text)
    qr_code = db.Column(db.String(100), nullable=False, unique=True)
    is_exit = db.Column(db.Boolean, nullable=False, default=False)

    exit_logs = db.relationship('VehicleExitLog', backref='vehicle', lazy=True)

class VehicleExitLog(db.Model):
    __tablename__ = 'vehicle_exit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.Integer, db.ForeignKey('vehicles.id'), nullable=False)
    exit_time = db.Column(db.DateTime, nullable=False, default=current_time_wib)
    exit_image_path = db.Column(db.String(255), nullable=False)
    match_score = db.Column(db.Float, nullable=True)
    match_status = db.Column(db.String(20), nullable=False)

# APP SETUP 
def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True)

    # Konfigurasi koneksi PostgreSQL
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Konfigurasi upload folder
    app.config['UPLOAD_FOLDER_IN'] = 'static/images/vehicles/masuk'
    os.makedirs(app.config['UPLOAD_FOLDER_IN'], exist_ok=True)

    app.config['UPLOAD_FOLDER'] = 'static/images/vehicles/keluar'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Inisialisasi database dan Marshmallow
    db.init_app(app)
    ma.init_app(app)

    # Membuat tabel di database jika belum ada
    with app.app_context():
        db.create_all()
    
            
    @app.route('/')
    def hello():
        return 'Hello, Flask di Ubuntu!'

    def extract_yolo_features(result, model_labels):
        detections = result[0].boxes
        boxes = detections.xywh.cpu().numpy()
        scores = detections.conf.cpu().numpy()
        classes = detections.cls.cpu().numpy().astype(int)

        label_counts = {label: 0 for label in model_labels.values()}
        confidence_per_label = {label: [] for label in model_labels.values()}

        widths = []
        heights = []

        for i, box in enumerate(boxes):
            class_idx = classes[i]
            label = model_labels[class_idx]
            confidence = scores[i]

            label_counts[label] += 1
            confidence_per_label[label].append(confidence)

            widths.append(box[2])
            heights.append(box[3])

        # Buat vektor fitur
        feature_vector = []
        for label in model_labels.values():
            feature_vector.append(label_counts[label])
            avg_conf = np.mean(confidence_per_label[label]) if confidence_per_label[label] else 0
            feature_vector.append(avg_conf)

        # Tambahkan ukuran rata-rata bbox
        feature_vector.append(np.mean(widths) if widths else 0)
        feature_vector.append(np.mean(heights) if heights else 0)

        return np.array(feature_vector)

    def convert_uploaded_file_to_cv2(image_file):
        image = Image.open(image_file).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Fungsi Koreksi OCR
    def correct_ocr(text):
        print(text)
        return text.replace("7", "1")

    # Fungsi OCR Plat Nomor
    def detect_license_plate_text(image_file, model_path):
        # Konversi file upload ke format OpenCV
        image = convert_uploaded_file_to_cv2(image_file)

        # Simpan ke file temporer agar YOLO bisa baca
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "temp_uploaded.jpg")        
        cv2.imwrite(temp_path, image)

        # Deteksi dengan YOLO
        model = YOLO(model_path)
        results = model(temp_path)

        detected_text = None

        print("Detected objects:")
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  # pastikan ambil nilai int dari tensor
                label = model.names[class_id] if hasattr(model, "names") else "Unknown"

                if label.lower() == "plat nomor":
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    print(f"Cropping coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                    plate_image = image[y1:y2, x1:x2]

                    try:
                        reader = easyocr.Reader(["en", "id"], gpu=False)
                        ocr_results = reader.readtext(plate_image)

                        for detection in ocr_results:
                            text = detection[1]
                            confidence = detection[2]
                            print(f"Detected text: {text}, confidence: {confidence}")

                            if confidence >= 0.55:
                                detected_text = correct_ocr(text) if 'correct_ocr' in globals() else text
                                break
                    except Exception as e:
                        print(f"OCR processing failed: {e}")
                    break

        return detected_text
    
    @app.route('/vehicle-entry', methods=['POST'])
    def upload_entry():
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image = request.files['image']
        feature = request.form.get('feature')
        vehicle_type = request.form.get('vehicle_type')
        local_save_path = request.form.get('entry_image_path')

        license_plate  = detect_license_plate_text(
            image_file=image,
            model_path="yolov8.pt"
        )

        print("License plate text:", license_plate)

        if not image or not license_plate:
            return jsonify({'error': 'Missing data'}), 400

         # buat nama file
        timestamp = datetime.now(pytz.timezone("Asia/Jakarta")).strftime('%Y%m%d%H%M%S')

        # simpan ke database
        qr_code_value = f"{license_plate}_{timestamp}"
        vehicle = Vehicle(
            license_plate=license_plate,
            vehicle_type=vehicle_type,
            entry_image_path=local_save_path,
            qr_code=qr_code_value,
            feature=feature  # simpan vektor fitur YOLO
            )
        db.session.add(vehicle)
        db.session.commit()

        return jsonify({
            'message': 'Image and data saved successfully',
            'license_plate': license_plate,
            'qr_code': qr_code_value,
            'entry_image_path': local_save_path
        }), 200


    @app.route('/vehicle-exit', methods=['POST'])
    def upload_exit():

        qr_code = request.form.get('qr_code')

        if not qr_code:
            return jsonify({'error': 'Missing data'}), 400

        # Cari kendaraan berdasarkan QR code
        vehicle = Vehicle.query.filter_by(qr_code=qr_code).first()
        if not vehicle:
            return jsonify({'error': 'Vehicle not found'}), 404
        
        # Step 2: QR code ditemukan, jalankan exit_cam.py
        # try:
        #     result = subprocess.run(['python', 'exit_cam.py'], check=True, capture_output=True, text=True)
        #     print('exit_cam.py output:', result.stdout)
        # except subprocess.CalledProcessError as e:
        #     return jsonify({'error': f'Failed to run exit_cam.py: {e.stderr}'}), 500

        image = request.files['image']
        feature = request.form.get('feature')
        vehicle_type = request.form.get('vehicle_type')
        local_save_path = request.form.get('exit_image_path')
        
        if not image:
            return jsonify({'error': 'Missing image'}), 400
        
        license_plate = detect_license_plate_text(
            image_file=image,
            model_path="yolov8.pt"
        )
        
        import json

        # Mapping singkatan ke label lengkap
        type_aliases = {'M': 'Motor', 'MB': 'Mobil', 'P': 'Plat Nomor'}

        # Exit Data Vehicle
        vehicle_type_exit = vehicle_type  # Bisa berupa string atau list JSON
        if isinstance(vehicle_type_exit, str):
            try:
                vehicle_type_exit = json.loads(vehicle_type_exit)
            except json.JSONDecodeError:
                vehicle_type_exit = [vehicle_type_exit]  # Jika hanya satu label, bungkus jadi list

        # Konversi singkatan ke label lengkap
        vehicle_type_exit = [type_aliases.get(v, v) for v in vehicle_type_exit]
        license_plate_exit = license_plate
        features2 = feature

        # Entry Data Vehicle
        license_plate_entry = vehicle.license_plate
        vehicle_type_entry = vehicle.vehicle_type  # Berformat string JSON
        features1_str = vehicle.feature
        print("vehicle_type_entry (raw):", vehicle_type_entry)

        # Decode JSON string
        try:
            vehicle_type_entry = json.loads(vehicle_type_entry)
        except json.JSONDecodeError:
            vehicle_type_entry = [vehicle_type_entry]

        # Konversi singkatan ke label lengkap
        vehicle_type_entry = [type_aliases.get(v, v) for v in vehicle_type_entry]

        # Mapping label ke angka
        label_map = {"Mobil": 0, "Motor": 1, "Plat Nomor": 2}

        try:
            labels_entry_numeric = [label_map[label] for label in vehicle_type_entry]
            labels_exit_numeric = [label_map[label] for label in vehicle_type_exit]
        except KeyError as e:
            print(f"[ERROR] Label tidak ditemukan dalam label_map: {e}")
            labels_entry_numeric, labels_exit_numeric = [], []  # Atau lakukan penanganan lain

        features1 = np.array(json.loads(features1_str))  # bentuk array float
        features2 = np.array(json.loads(features2))  # bentuk array float

        # features2 = features2

        # Cocokan Kendaraan masuk dan keluar
        is_match = False
        if license_plate_exit == license_plate_entry and vehicle_type_exit == vehicle_type_entry:
            print("Cocok! Data kendaraan sesuai.")
            is_match = True
        else:
            print("Data tidak cocok. Pemeriksaan manual dibutuhkan.")


        # Image Matching logic
        # Gabungkan fitur + label
        # features1 = np.hstack([features1_str.flatten(), labels_entry_numeric])
        # features2 = np.hstack([features2.flatten(), labels_exit_numeric[:len(features2)]])  # Ambil sesuai jumlah boxes
        try:
            features1_str = json.loads(features1_str)
        except json.JSONDecodeError:
            print("[ERROR] Gagal parse JSON dari features1_str")
            features1_str = []

                # Pastikan features1_str dan features2 adalah numpy array
        features1_array = np.array(features1_str, dtype=np.float32)
        features2_array = np.array(features2, dtype=np.float32)

        # Flatten fitur
        features1_flat = features1_array.flatten()
        features2_flat = features2_array.flatten()

        # Validasi dan pemotongan label agar sesuai jumlah box (kalau perlu)
        labels_entry_numeric = np.array(labels_entry_numeric, dtype=np.float32)
        labels_exit_numeric = np.array(labels_exit_numeric, dtype=np.float32)

        # Sesuaikan panjang label dengan panjang fitur
        num_boxes_entry = len(features1_flat)   
        num_boxes_exit = len(features2_flat)

        labels_entry_trimmed = labels_entry_numeric[:num_boxes_entry]
        labels_exit_trimmed = labels_exit_numeric[:num_boxes_exit]

        # Gabungkan fitur + label
        features1_combined = np.hstack([features1_flat, labels_entry_trimmed])
        features2_combined = np.hstack([features2_flat, labels_exit_trimmed])

        # Hitung cosine similarity
        similarity = cosine_similarity([features1_combined], [features2_combined])[0][0]

        # Bulatkan skor similarity
        match_score = round(float(similarity), 3)

        # Tentukan status kecocokan
        match_status = "matched" if match_score >= 0.9 else "not_matched"

        # Simpan log keluar
        exit_log = VehicleExitLog(
            vehicle_id=vehicle.id,
            exit_image_path=local_save_path,
            match_score=match_score,
            match_status=match_status
        )
        db.session.add(exit_log)

        # Tandai kendaraan sudah keluar
        vehicle.is_exit = True
        db.session.commit()

        # Kembalikan respons ke client
        return jsonify({
            'message': 'Exit data saved successfully',
            'license_plate': vehicle.license_plate,
            'exit_image_path': local_save_path,
            'match_score': match_score,
            'match_status': match_status,
            'vehicle_match': is_match  # Tambahan info cocok/tidak
        }), 200

    return app

# MAIN
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)

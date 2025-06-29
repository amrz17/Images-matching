import os
import cv2
import json
import requests
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO

# Inisialisasi model dan konfigurasi hanya sekali saat file diimport
load_dotenv()
camera_ip = os.getenv("CAMERA_IP")

# Load YOLO model
model = YOLO("yolov8.pt")
labels = model.names

save_dir = "vehicle_detections/vehicle_entry"
os.makedirs(save_dir, exist_ok=True)

def get_detected_objects_array(boxes, class_ids, scores):
    labels_map = {
        0: "Mobil",
        1: "Motor",
        2: "Plat Nomor"
    }

    features = []
    detected_labels = []

    for i, box in enumerate(boxes):
        class_idx = int(class_ids[i])
        label = labels_map.get(class_idx, "Unknown")
        confidence = float(scores[i])

        x, y, w, h = map(float, box)
        x2 = x + w
        y2 = y + h

        features.append([x, y, x2, y2])
        detected_labels.append(label)

    return features, detected_labels

def detect_objects_from_frame(frame):
    results = model.predict(frame)
    boxes_raw = results[0].boxes
    boxes = boxes_raw.xywh.cpu().numpy()
    scores = boxes_raw.conf.cpu().numpy()
    class_ids = boxes_raw.cls.cpu().numpy().astype(int).tolist()
    annotated_frame = results[0].plot()

    features, detected_labels = get_detected_objects_array(boxes, class_ids, scores)

    # Serialize
    feature_json = json.dumps(features)
    labels_json = json.dumps(detected_labels)

    # Tentukan label utama untuk nama file
    primary_label = "Unknown"
    for label in detected_labels:
        if label in ["Mobil", "Motor"]:
            primary_label = label
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = "original_image.jpg"
    local_save_path = os.path.join(save_dir, f"{primary_label}_{timestamp}.jpg")
    annotated_filename = os.path.join(save_dir, f"{primary_label}_{timestamp}_annotated.jpg")

    # Simpan gambar
    # cv2.imwrite(image_filename, frame)
    # cv2.imwrite(local_save_path, frame)
    # cv2.imwrite(annotated_filename, annotated_frame)

    print(f"[+] Gambar disimpan di: {local_save_path}")

    # # Kirim ke API jika memenuhi syarat
    # if (0 in class_ids and 2 in class_ids) or (1 in class_ids and 2 in class_ids):
    #     try:
    #         with open(image_filename, 'rb') as img_file:
    #             files = {'image': img_file}
    #             data = {
    #                 'feature': feature_json,
    #                 'vehicle_type': labels_json,
    #                 'entry_image_path': local_save_path
    #             }
    #             response = requests.post("http://localhost:5000/vehicle-entry", files=files, data=data)
    #             print(f"[+] API Response: {response.status_code}")
    #     except requests.exceptions.RequestException as e:
    #         print(f"[!] Gagal mengirim ke API: {e}")
    # else:
    #     print("Kombinasi class yang diinginkan tidak terdeteksi.")

    return annotated_frame, detected_labels

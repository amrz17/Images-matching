import os
import cv2
import json
import requests
import numpy as np
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from dotenv import load_dotenv
from datetime import datetime

from ultralytics import YOLO


# Load .env
load_dotenv()
camera_ip = os.getenv("CAMERA_IP")

# Load model YOLO
detection_model = YOLO("yolov8.pt")

# Load model ResNet50
model_resnet = models.resnet50(weights=None)
model_resnet.load_state_dict(torch.load('resnet50.pth'))
model_resnet.eval() # model evaluasi

# Hilangkan bagian klasifikasi akhir (ambil fitur saja)
feature_extractor = torch.nn.Sequential(*list(model_resnet.children())[:-1])

# Resize and preprocessing for ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize 224x224
    transforms.ToTensor(),  # Ubah format ke tensor pytorch
    transforms.Normalize(  # sesuai mean/std ImageNet # Normalisasi channel warna
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Buat folder untuk simpan hasil frame
save_dir = r"vehicle_detections/vehicle_entry/"
os.makedirs(save_dir, exist_ok=True)

def safe_json_serialize(obj):
    # Konversi objek ke format yang bisa diserialisasi JSON
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()  # Konversi np.float32 ke float Python
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    return obj

# Ekstrak fitur with ResNet
def extract_features(img_array):
    # Validasi input
    if not isinstance(img_array, np.ndarray):
        raise ValueError("Input harus berupa numpy array")
    
    # Konversi grayscale ke RGB 
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif len(img_array.shape) == 3:
        # Konversi BGR ke RGB untuk OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Format gambar tidak didukung")
    
    try:
        # Preprocess gambar
        # Konversi ke PIL Image
        img_array = Image.fromarray(img_array)

        input_tensor = transform(img_array)
        input_batch = input_tensor.unsqueeze(0)  # Tambah dimensi batch
        
        # Ekstrak fitur
        with torch.no_grad():
            features = feature_extractor(input_batch)  # Shape: [1, 2048, 7, 7]
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # Shape: [1, 2048, 1, 1]
            features = features.flatten(start_dim=1)  # Shape: [1, 2048]
        
        return features
    
    except Exception as e:
        print(f"Error saat ekstraksi fitur: {str(e)}")
        return None

def detect_and_crop_object(input_image, save_dir="output"):
    detection_model = YOLO("yolov8.pt")
    if isinstance(input_image, str):
        img = cv2.imread(input_image)
        if img is None:
            print(f"Error: Gagal memuat gambar dari {input_image}")
            return None, None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = img.copy()
    elif isinstance(input_image, np.ndarray):
        img_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        frame = input_image.copy()
    else:
        print("Error: Input harus berupa path gambar atau numpy array")
        return None, None

    labels_map = {
        0: "Mobil",
        1: "Motor",
        2: "Plat Nomor"
    }

    results = detection_model(img_rgb)
    boxes_data = results[0].boxes

    boxes = boxes_data.xyxy.cpu().numpy()
    classes = boxes_data.cls.cpu().numpy().astype(int)
    confidences = boxes_data.conf.cpu().numpy()

    features_list = []
    detected_labels = []

    for box, cls_id, conf in zip(boxes, classes, confidences):
        if cls_id in [0, 1]:  # Mobil atau Motor
            label = labels_map.get(cls_id, "Unknown")
            x1, y1, x2, y2 = map(int, box)
            cropped_object = img_rgb[y1:y2, x1:x2]

            features = extract_features(cropped_object)
            if features is not None:
                features_list.append(features)
                detected_labels.append(label)

    if not features_list:
        print("Tidak ada objek kendaraan yang terdeteksi.")
        return None, None

    features_array = np.array(features_list)
    feature_json = json.dumps(features_array.tolist())
    vehicle_types_json = json.dumps(detected_labels)

    print("Tipe labels:", type(detected_labels))
    print("Feature shape:", features_array.shape)
    print("Label:", vehicle_types_json)

    primary_label = "Unknown"
    for label in detected_labels:
        if label in ["Mobil", "Motor"]:
            primary_label = label
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    original_filename = os.path.join(save_dir, f"{primary_label}_{timestamp}.jpg")
    annotated_filename = os.path.join(save_dir, f"{primary_label}_{timestamp}_annotated.jpg")

    cv2.imwrite(original_filename, frame)
    annotated_frame = frame.copy()  # Gantilah ini dengan frame yang berisi bounding box jika ada
    cv2.imwrite(annotated_filename, annotated_frame)

    print(f"[+] Gambar asli disimpan di: {original_filename}")

    # Kirim hanya jika deteksi mobil/motor dan plat nomor
    detected_class_ids = set(classes)
    if (0 in detected_class_ids and 2 in detected_class_ids) or (1 in detected_class_ids and 2 in detected_class_ids):
        try:
            with open(original_filename, 'rb') as img_file:
                files = {'image': img_file}
                data = {
                    'feature': feature_json,
                    'vehicle_type': vehicle_types_json,
                    'entry_image_path': original_filename
                }
                response = requests.post("http://localhost:5000/vehicle-entry", files=files, data=data)
                print(f"[+] API Response: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[!] Gagal mengirim ke API: {e}")
    else:
        print("Kombinasi class yang diinginkan tidak terdeteksi.")

    # Bersihkan
    return annotated_frame, vehicle_types_json

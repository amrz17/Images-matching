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

# Buka kamera
cap = cv2.VideoCapture(camera_ip)

if not cap.isOpened():
    print("Gagal membuka kamera!")
    exit()

ret, frame = cap.read()
if not ret:
    print("Gagal membaca frame dari kamera!")
    cap.release()
    exit()

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

# Proses deteksi
results = detection_model.predict(frame)
boxes = results[0].boxes
class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
annotated_frame = results[0].plot()

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

# Detection with yolo
def detect_and_crop_object(image_path, detection_model):
    # Load image dari path
    # Handle both file path and numpy array inputs
    if isinstance(image_path, str):
        # Jika input adalah path file
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Gagal memuat gambar dari {image_path}")
            return None, None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path, np.ndarray):
        # Jika input sudah berupa numpy array (frame dari kamera)
        img_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    else:
        print("Error: Input harus berupa path gambar atau numpy array")
        return None, None

    labels_map = {
        0: "Mobil",
        1: "Motor",
        2: "Plat Nomor"
    }

    # Deteksi objek
    results = detection_model(img_rgb)
    boxe = results[0].boxes

    # Get all detected objects
    features_list = []
    class_ids = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        classes = result.boxes.cls.cpu().numpy()  # Get class IDs
        confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
        class_ids = boxe.cls.cpu().numpy().astype(int).tolist()

        for box, cls_id, conf in zip(boxes, classes, confidences):
            # Only process class 1 (Mobil) or 2 (Motor)
            if cls_id in [0, 1]:  
                label = labels_map.get(cls_id, "Unknown")
                x1, y1, x2, y2 = map(int, box)
                cropped_object = img_rgb[y1:y2, x1:x2]  # Crop detected object
                
                # Ekstrak fitur
                features = extract_features(cropped_object)
                if features is not None:
                    features_list.append(features)
                # cropped_objects.append(cropped)
                class_ids.append(label)

    return features, class_ids

features, detected_labels = detect_and_crop_object(frame, detection_model)
print("Tipe labels:", type(detected_labels))
print("Labels:", detected_labels)
print("Tipe Features:", type(features))
print("feature", features.shape)

feature = features.tolist()  # ubah ke list Python 
feature_json = json.dumps(feature)  # baru serialisasi ke JSON

vehicle_types_json = json.dumps(detected_labels)

print("Label", vehicle_types_json)


# Tampilkan annotated frame ke layar
# cv2.imshow("Deteksi Kamera", annotated_frame)
# cv2.waitKey(1000)

# Tentukan label utama untuk nama file
primary_label = "Unknown"
for label in detected_labels:
    if label in ["Mobil", "Motor"]:
        primary_label = label
        break

# Simpan frame asli (tanpa bounding box)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_filename = "original_image.jpg"
local_save_path = os.path.join(save_dir, f"{primary_label}_{timestamp}.jpg")
annotated_filename = os.path.join(save_dir, f"{primary_label}_{timestamp}_annotated.jpg")


cv2.imwrite(image_filename, frame)         # untuk dikirim ke API
cv2.imwrite(local_save_path, frame)        # simpan ke penyimpanan lokal
cv2.imwrite(annotated_filename, annotated_frame)
print(f"[+] Gambar asli disimpan di: {local_save_path}")

# # Kirim hanya jika class 0 dan 2, atau 1 dan 2 terdeteksi
if (0 in class_ids and 2 in class_ids) or (1 in class_ids and 2 in class_ids):
    try:
        with open(image_filename, 'rb') as img_file:
            files = {'image': img_file}
            data = {
                'feature': feature_json,
                'vehicle_type': vehicle_types_json,
                'entry_image_path': local_save_path
            }
            response = requests.post("http://localhost:5000/vehicle-entry", files=files, data=data)
            print(f"[+] API Response: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[!] Gagal mengirim ke API: {e}")
else:
    print("Kombinasi class yang diinginkan tidak terdeteksi.")

# Bersihkan
cap.release()
cv2.destroyAllWindows()

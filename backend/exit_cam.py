import cv2
import os
import sys
import requests
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO
import json

# Load .env
load_dotenv()
camera_ip = os.getenv("CAMERA_IP")

# Load model YOLO
model = YOLO("yolov8.pt")

# Buat folder untuk simpan hasil frame
save_dir = "vehicle_detections/vehicle_exit"
os.makedirs(save_dir, exist_ok=True)

# Ambil QR code dari command line argument atau input user
qr_code = ""
if len(sys.argv) > 1:
    qr_code = sys.argv[1]
else:
    qr_code = input("Masukkan QR code: ")

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

# Proses deteksi
results = model.predict(frame)
boxes = results[0].boxes
boxes1 = boxes.xywh.cpu().numpy()  # Koordinat bounding box (x, y, w, h)
scores1 = boxes.conf.cpu().numpy()  # Skor kepercayaan
class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
labels = model.names  # Daftar nama label dari model
annotated_frame = results[0].plot()

def get_detected_objects_array(boxes1, class_ids, scores1):
    labels_map = {
        0: "Mobil",
        1: "Motor",
        2: "Plat Nomor"
    }

    features = []
    detected_labels = []

    for i, box in enumerate(boxes1):
        class_idx = int(class_ids[i])
        label = labels_map.get(class_idx, "Unknown")
        confidence = float(scores1[i])
        
        x, y, w, h = map(float, box)
        x2 = x + w
        y2 = y + h

        print(f"Object: {label} - Confidence: {confidence:.2f}")
        print(f"Bounding Box (x1, y1, x2, y2): {x}, {y}, {x2}, {y2}")

        features.append([x, y, x2, y2])
        detected_labels.append(label)
    
    # Return all detected labels instead of just one
    return features, detected_labels

features, detected_labels = get_detected_objects_array(boxes1, class_ids, scores1)

# Convert to JSON for API
features_json = json.dumps(features)
vehicle_types_json = json.dumps(detected_labels)

# vehicle_types_json = detected_labels[:5]


# vehicle_types_json = json.loads(detected_labels)[:5]  # Get first 5 items from the parsed JSON


print("Labels", detected_labels)
print("feature", features_json)

# Tampilkan annotated frame ke layar
cv2.imshow("Deteksi Kamera", annotated_frame)
cv2.waitKey(1000)

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

# Kirim hanya jika class 0 dan 2, atau 1 dan 2 terdeteksi
if (0 in class_ids and 2 in class_ids) or (1 in class_ids and 2 in class_ids):
    try:
        with open(image_filename, 'rb') as img_file:
            # Perbaikan: Ubah format files untuk content-type yang benar
            files = {'image': ('image.jpg', img_file, 'image/jpeg')}
            
            # Perbaikan: Tambahkan qr_code dan ubah entry_image_path menjadi exit_image_path
            data = {
                'qr_code': qr_code,              # Tambahkan QR code
                'feature': features_json,
                'vehicle_type': vehicle_types_json,  # Kirim semua label yang terdeteksi
                'exit_image_path': local_save_path   # Ubah nama parameter yang benar
            }
            
            print("Mengirim data:")
            print(f"- QR Code: {qr_code}")
            print(f"- Vehicle Types: {detected_labels}")
            
            response = requests.post("http://localhost:5000/vehicle-exit", files=files, data=data)
            print(f"[+] API Response: {response.status_code}")
            
            # Tampilkan respons jika tersedia
            if response.status_code == 200:
                print(f"[+] Respons API: {response.json()}")
            else:
                print(f"[-] Error: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"[!] Gagal mengirim ke API: {e}")
else:
    print("Kombinasi class yang diinginkan tidak terdeteksi.")

# Bersihkan
cap.release()
cv2.destroyAllWindows()
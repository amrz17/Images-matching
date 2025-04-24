import cv2
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO
import json
import pytz

# Load .env
load_dotenv()
camera_ip = os.getenv("CAMERA_IP")

# Load model YOLO
model = YOLO("yolov8.pt")

# Buat folder untuk simpan hasil frame
save_dir = "vehicle_detections/vehicle_exit"
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
        0: "motor",
        1: "mobil",
        2: "plat"
    }

    results = []
    label = []
    for i, box in enumerate(boxes1): 
        class_id = float(class_ids[i])
        label = labels_map.get(class_id, "unknown")
        score = float(scores1[i])
        x, y, w, h = map(float, box)

        # Hitung x2 dan y2 dari w dan h
        x2 = x + w
        y2 = y + h

        result = [class_id, score, x, y, w, h, x2, y2]
        results.append(result)

        if class_id in [0, 1, 2]:  # motor dan mobil
            label.append(label)

    return results, label

feature, label = get_detected_objects_array(boxes1, class_ids, scores1)
print("feature = ", feature)
print("labels = ", label)

feature_json = json.dumps(feature, indent=2)
#label = label[0]

print('feature', feature_json)

# Tampilkan annotated frame ke layar
cv2.imshow("Deteksi Kamera", annotated_frame)
cv2.waitKey(1000)

# Simpan frame asli (tanpa bounding box)
timestamp = datetime.now(pytz.timezone("Asia/Jakarta")).strftime('%Y%m%d%H%M%S')
image_filename = "original_image.jpg"
local_save_path = os.path.join(save_dir, f"original_{timestamp}.jpg")


cv2.imwrite(image_filename, frame)         # untuk dikirim ke API
cv2.imwrite(local_save_path, frame)        # simpan ke penyimpanan lokal
print(f"[+] Gambar asli disimpan di: {local_save_path}")

# Kirim hanya jika class 0 dan 2, atau 1 dan 2 terdeteksi
if (0 in class_ids and 2 in class_ids) or (1 in class_ids and 2 in class_ids):
    try:
        with open(image_filename, 'rb') as img_file:
            files = {'image': img_file}
            data = {
                'feature': feature_json,
                'vehicle_type': label
            }
            response = requests.post("http://localhost:5000/upload-entry", files=files, data=data)
            print(f"[+] API Response: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[!] Gagal mengirim ke API: {e}")
else:
    print("Kombinasi class yang diinginkan tidak terdeteksi.")

# Bersihkan
cap.release()
cv2.destroyAllWindows()

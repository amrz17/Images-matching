from flask import Flask, jsonify, Response, request, render_template
import requests
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
import ast


# import qrcode
# import qrcode

import torch
import torchvision.models as models
import torchvision.transforms as transforms


load_dotenv()  # Ini akan membaca file .env
camera_ip = os.getenv("CAMERA_IP")

# Load model YOLO sekali saja saat startup
model = YOLO("yolov8.pt")
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
    vehicle_type = db.Column(db.String(50), nullable=True)
    feature = db.Column(db.Text)
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

    # Buat folder untuk simpan hasil frame kendaraan keluar
    save_dir = r"vehicle_detections/vehicle_exit/"
    save_dir_entry = r"vehicle_detections/vehicle_entry/"
    save_dir_qr = "vehicle_detections/qr_code"
    os.makedirs(save_dir, exist_ok=True)

    # Inisialisasi database dan Marshmallow
    db.init_app(app)
    ma.init_app(app)

    # Membuat tabel di database jika belum ada
    with app.app_context():
        db.create_all()
    
            
    @app.route('/')
    def hello():
        return render_template('index.html')

    # def get_detected_objects_array(boxes1, class_ids, scores1):
    #     labels_map = {
    #         0: "Mobil",
    #         1: "Motor",
    #         2: "Plat Nomor"
    #     }
    #     features = []
    #     detected_labels = []
    #     for i, box in enumerate(boxes1):
    #         class_idx = int(class_ids[i])
    #         label = labels_map.get(class_idx, "Unknown")
    #         confidence = float(scores1[i])

    #         x, y, w, h = map(float, box)
    #         x2 = x + w
    #         y2 = y + h

    #         print(f"Object: {label} - Confidence: {confidence:.2f}")
    #         print(f"Bounding Box (x1, y1, x2, y2): {x}, {y}, {x2}, {y2}")

    #         features.append([x, y, x2, y2])
    #         detected_labels.append(label)

    #     return features, detected_labels
    

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


    # def buat_qr_code(data):
    #     # Membuat QR Code
    #     qr = qrcode.QRCode(
    #         version=1,
    #         error_correction=qrcode.constants.ERROR_CORRECT_L,
    #         box_size=10,
    #         border=4,
    #     )
    #     qr.add_data(data)
    #     qr.make(fit=True)

    #     # Membuat gambar dari QR code
    #     img = qr.make_image(fill_color="black", back_color="white")

    #     # Simpan frame asli (tanpa bounding box)
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     image_filename = "qr.jpg"
    #     local_save_path = os.path.join(save_dir_qr, f"{image_filename}_{timestamp}.jpg")
    #     cv2.imwrite(local_save_path, img)        # simpan ke penyimpanan lokal
    #     print(f"QR Code berhasil disimpan sebagai {local_save_path}")


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

                        # Filter dan urutkan hasil berdasarkan posisi horizontal (x kiri atas)
                        filtered_results = [
                            (detection[0][0][0], correct_ocr(detection[1]), detection[2])
                            for detection in ocr_results
                            if detection[2] > 0.8  # confidence > 0.8
                        ]

                        # Urutkan dari kiri ke kanan berdasarkan koordinat x
                        filtered_results.sort(key=lambda x: x[0])

                        # Satukan semua hasil dalam satu string
                        detected_text = " ".join([item[1] for item in filtered_results])

                        print(f"Final Corrected OCR Result (confidence > 0.8): {detected_text}")
                    except Exception as e:
                        print(f"OCR processing failed: {e}")
                    break

        return detected_text


    # def detect_and_crop_object(image_path, detection_model):
    #     # Load image dari path
    #     # Handle both file path and numpy array inputs
    #     if isinstance(image_path, str):
    #         # Jika input adalah path file
    #         img = cv2.imread(image_path)
    #         if img is None:
    #             print(f"Error: Gagal memuat gambar dari {image_path}")
    #             return None, None
    #         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     elif isinstance(image_path, np.ndarray):
    #         # Jika input sudah berupa numpy array (frame dari kamera)
    #         img_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    #     else:
    #         print("Error: Input harus berupa path gambar atau numpy array")
    #         return None, None

    #     labels_map = {
    #         0: "Mobil",
    #         1: "Motor",
    #         2: "Plat Nomor"
    #     }

    #     # Deteksi objek
    #     results = detection_model(img_rgb)
    #     boxe = results[0].boxes

    #     # Get all detected objects
    #     features_list = []
    #     class_ids = []

    #     for result in results:
    #         boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
    #         classes = result.boxes.cls.cpu().numpy()  # Get class IDs
    #         confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
    #         class_ids = boxe.cls.cpu().numpy().astype(int).tolist()

    #         for box, cls_id, conf in zip(boxes, classes, confidences):
    #             # Only process class 1 (Mobil) or 2 (Motor)
    #             if cls_id in [0, 1]:  
    #                 label = labels_map.get(cls_id, "Unknown")
    #                 x1, y1, x2, y2 = map(int, box)
    #                 # Crop detected object
    #                 cropped_object = img_rgb[y1:y2, x1:x2]  
                    
    #                 # Ekstrak fitur
    #                 features = extract_features(cropped_object)
    #                 if features is not None:
    #                     features_list.append(features)
    #                 # cropped_objects.append(cropped)
    #                 class_ids.append(label)

    #     return features, class_ids

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

    @app.route('/vehicle-entry', methods=['POST'])
    def upload_entry():
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image = request.files['image']
        feature = request.form.get('feature')
        vehicle_type = request.form.get('vehicle_type')
        local_save_path = request.form.get('entry_image_path')

        # # Untuk real-time frame
        # image = request.files['image']
# 
        license_plate  = detect_license_plate_text(
            image_file=image,
            model_path="yolov8.pt"
        )

        # file_bytes = np.frombuffer(image.read(), dtype=np.uint8)
        # image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # <-- hasilnya sama tipe seperti 'frame'
        # print(type(image))

        # # Mulai dari sini api real-time di app
        # # Baca isi file dan konversi ke np.ndarray (mirip frame webcam)
        # license_plate = license_plate 
        # print("License plate text:", license_plate)

        # results = model.predict(image)
        # boxes = results[0].boxes
        # boxes1 = boxes.xywh.cpu().numpy()
        # scores1 = boxes.conf.cpu().numpy()
        # class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        # labels = model.names
        # annotated_frame = results[0].plot()

        # features, detected_labels = get_detected_objects_array(boxes1, class_ids, scores1)
        # features_json = json.dumps(features)
        # vehicle_types_json = json.dumps(detected_labels)

        # features, detected_labels = detect_and_crop_object(image, model)
        # print("Tipe labels:", type(detected_labels))
        # print("feature", features.shape)

        # feature = features.tolist()  # ubah ke list Python 
        # features_json = json.dumps(feature)  # baru serialisasi ke JSON

        # vehicle_type = json.dumps(detected_labels)

        # print("Label", vehicle_types_json)

        # primary_label = "Unknown"
        # for label in detected_labels:
        #     if label in ["Mobil", "Motor"]:
        #         primary_label = label
        #         break
        
        # # Simpan frame asli (tanpa bounding box)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # image_filename = "original_image.jpg"
        # local_save_path = os.path.join(save_dir_entry, f"{primary_label}_{timestamp}.jpg")
        # annotated_filename = os.path.join(save_dir_entry, f"{primary_label}_{timestamp}_annotated.jpg")

        # cv2.imwrite(image_filename, image)         # untuk dikirim ke API
        # cv2.imwrite(local_save_path, image)        # simpan ke penyimpanan lokal
        # cv2.imwrite(annotated_filename, annotated_frame)
        # print(f"[+] Gambar asli disimpan di: {local_save_path}")

        # # akhir dari sini untuk api real-time di app

        if not image or not license_plate:
            return jsonify({'error': 'Missing data'}), 400

         # buat nama file
        timestamp = datetime.now(pytz.timezone("Asia/Jakarta")).strftime('%Y%m%d%H%M%S')

        # simpan ke database
        qr_code_value = f"{license_plate}_{timestamp}"

        # # Simpan frame asli (tanpa bounding box)
        # image_qr = "qr_code.jpg"
        # local_save_qr = os.path.join(save_dir_qr, f"{qr_code_value}.jpg")

        # cv2.imwrite(image_qr, qr)         # untuk dikirim ke API
        # cv2.imwrite(local_save_qr, qr)        # simpan ke penyimpanan lokal
        # print(f"[+] Gambar asli disimpan di: {local_save_qr}")
        
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
    
    @app.route('/vehicle-entry', methods=['GET'])
    def get_vehicle_entry():
        return jsonify({
            "message": "Image and data saved successfully",
            "entry_image_path": "static/images/vehicle_123.jpg"
    })


    # Endpoint untuk menerima dan langsung memberikan QR code (tanpa menyimpan)
    @app.route('/get-latest-qr', methods=['POST'])
    def get_latest_qr():
        data = request.get_json()
        qr_code = data.get("qr_code")
        
        if not qr_code:
            return jsonify({"error": "qr_code tidak ditemukan"}), 400

        # Mulai proses deteksi
        cap = cv2.VideoCapture(camera_ip)
        if not cap.isOpened():
            return jsonify({"error": "Gagal membuka kamera"}), 500

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return jsonify({"error": "Gagal membaca frame"}), 500

        # Proses deteksi
        results = detection_model.predict(frame)
        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        annotated_frame = results[0].plot()

        features, detected_labels = detect_and_crop_object(frame, detection_model)

        print("Tipe labels:", type(detected_labels))
        print("Labels:", detected_labels)
        detected_labels = [item for item in detected_labels if isinstance(item, str)]
        print("Tipe Features:", type(features))
        print("feature", features.shape)

        feature = features.tolist()  # ubah ke list Python 
        features_json = json.dumps(feature)  # baru serialisasi ke JSON

        vehicle_types_json = json.dumps(detected_labels)

        print("Label", vehicle_types_json)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = "original_exit_image.jpg"
        local_save_path = os.path.join(save_dir, f"{image_filename}_{timestamp}.jpg")
        annotated_filename = os.path.join(save_dir, f"{image_filename}_{timestamp}_annotated.jpg")

        cv2.imwrite(image_filename, frame)
        cv2.imwrite(local_save_path, frame)
        cv2.imwrite(annotated_filename, annotated_frame)

        if (0 in class_ids and 2 in class_ids) or (1 in class_ids and 2 in class_ids):
            try:
                with open(image_filename, 'rb') as img_file:
                    files = {'image': ('image.jpg', img_file, 'image/jpeg')}
                    data = {
                        'qr_code': qr_code,
                        'feature': features_json,
                        'vehicle_type': vehicle_types_json,
                        'exit_image_path': local_save_path
                    }
                    response = requests.post("http://localhost:5000/vehicle-exit", files=files, data=data)

                    if response.status_code == 200:
                        return jsonify(response.json()), 200
                    else:
                        return jsonify({"error": response.text}), response.status_code

            except requests.exceptions.RequestException as e:
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"message": "Kombinasi class tidak sesuai"}), 200

        cap.release()
        cv2.destroyAllWindows()


    # Endpoint untuk menerima citra kendaraan keluar dan untuk
    # menjalankan fungsi pencocokan kendaraan masuk dan keluar
    @app.route('/vehicle-exit', methods=['POST'])
    def upload_exit():

        qr_code = request.form.get('qr_code')

        if not qr_code:
            return jsonify({'error': 'Missing data'}), 400

        # Cari kendaraan berdasarkan QR code
        vehicle = Vehicle.query.filter_by(qr_code=qr_code).first()
        if not vehicle:
            return jsonify({'error': 'Vehicle not found'}), 404
        
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
        
        # Exit Data Vehicle
        vehicle_type_exit = vehicle_type  # Bisa berupa string atau list JSON
        if isinstance(vehicle_type_exit, str):
            try:
                vehicle_type_exit = json.loads(vehicle_type_exit)
            except json.JSONDecodeError:
                vehicle_type_exit = [vehicle_type_exit]  # Jika hanya satu label, bungkus jadi list

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

        # Mapping label ke angka
        label_map = {"Mobil": 0, "Motor": 1, "Plat Nomor": 2}

        try:
            labels_entry_numeric = [label_map[label] for label in vehicle_type_entry]
            labels_exit_numeric = [label_map[label] for label in vehicle_type_exit]
        except KeyError as e:
            print(f"[ERROR] Label tidak ditemukan dalam label_map: {e}")
            labels_entry_numeric, labels_exit_numeric = [], []  # Atau lakukan penanganan lain

        features2 = np.array(json.loads(features2))  # bentuk array float

        # Cocokan Kendaraan masuk dan keluar
        is_match = False
        if license_plate_exit == license_plate_entry and vehicle_type_exit == vehicle_type_entry:
            print("Cocok! Data kendaraan sesuai.")
            is_match = True
        else:
            print("Data tidak cocok. Pemeriksaan manual dibutuhkan.")


        # Image Matching logic
        # Gabungkan fitur + label
        try:
            features1 = json.loads(features1_str)
            # features1 = features1.tolist()  # Convert numpy array to list
            # features1 = json.dumps(features1)  # Then convert to JSON stringk
        except json.JSONDecodeError:
            print("[ERROR] Gagal parse JSON dari features1_str")
            features1 = []

        # Pastikan features1_str dan features2 adalah numpy array
        features1_array = np.array(features1, dtype=np.float32)
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
        print(f"Match score: {match_score}")

        # Tentukan status kecocokan
        match_status = "matched" if match_score >= 0.8 else "not_matched"
        print(f"Data {match_status}")

        # Simpan log keluar
        exit_log = VehicleExitLog(
            vehicle_id=vehicle.id,
            vehicle_type=vehicle_type_exit,
            feature=feature,
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
        }), 200

    
    # Untuk GUI TkInter

    # Endpoint untuk menerima dan langsung memberikan QR code (tanpa menyimpan)
    @app.route('/get-qr-code', methods=['POST'])
    def get_qr_code():
        data = request.get_json()
        qr_code = data.get("qr_code")
        
        if not qr_code:
            return jsonify({"error": "qr_code tidak ditemukan"}), 400

        try:
                data = {
                    'qr_code': qr_code,
                }
                response = requests.post("http://localhost:5000/vehicle-exit", data=data)

                if response.status_code == 200:
                    return jsonify(response.json()), 200
                else:
                    return jsonify({"error": response.text}), response.status_code

        except requests.exceptions.RequestException as e:
            return jsonify({"error": str(e)}), 500

    # Endpoint untuk menerima citra kendaraan keluar dan untuk
    # menjalankan fungsi pencocokan kendaraan masuk dan keluar
    @app.route('/vehicle-exitv2', methods=['POST'])
    def upload_exitv2():

        qr_code = request.form.get('qr_code')

        if not qr_code:
            return jsonify({'error': 'Missing data'}), 400

        # Cari kendaraan berdasarkan QR code
        vehicle = Vehicle.query.filter_by(qr_code=qr_code).first()
        if not vehicle:
            return jsonify({'error': 'Vehicle not found'}), 404
        
        # Mulai proses deteksi
        cap = cv2.VideoCapture(camera_ip)
        if not cap.isOpened():
            return jsonify({"error": "Gagal membuka kamera"}), 500

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return jsonify({"error": "Gagal membaca frame"}), 500

        # Proses deteksi
        results = detection_model.predict(frame)
        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        annotated_frame = results[0].plot()

        features, detected_labels = detect_and_crop_object(frame, detection_model)

        print("Tipe labels:", type(detected_labels))
        print("Labels:", detected_labels)
        detected_labels = [item for item in detected_labels if isinstance(item, str)]
        print("Tipe Features:", type(features))
        print("feature", features.shape)

        feature = features.tolist()  # ubah ke list Python 
        features_json = json.dumps(feature)  # baru serialisasi ke JSON

        vehicle_types_json = json.dumps(detected_labels)

        print("Label", vehicle_types_json)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = "original_exit_image.jpg"
        local_save_path = os.path.join(save_dir, f"{image_filename}_{timestamp}.jpg")
        annotated_filename = os.path.join(save_dir, f"{image_filename}_{timestamp}_annotated.jpg")

        cv2.imwrite(image_filename, frame)
        cv2.imwrite(local_save_path, frame)
        cv2.imwrite(annotated_filename, annotated_frame)
        
        if not frame:
            return jsonify({'error': 'Missing image'}), 400
        
        license_plate = detect_license_plate_text(
            image_file=frame,
            model_path="yolov8.pt"
        )
        
        # Exit Data Vehicle
        vehicle_type_exit = vehicle_types_json  # Bisa berupa string atau list JSON
        if isinstance(vehicle_type_exit, str):
            try:
                vehicle_type_exit = json.loads(vehicle_type_exit)
            except json.JSONDecodeError:
                vehicle_type_exit = [vehicle_type_exit]  # Jika hanya satu label, bungkus jadi list

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

        # Mapping label ke angka
        label_map = {"Mobil": 0, "Motor": 1, "Plat Nomor": 2}

        try:
            labels_entry_numeric = [label_map[label] for label in vehicle_type_entry]
            labels_exit_numeric = [label_map[label] for label in vehicle_type_exit]
        except KeyError as e:
            print(f"[ERROR] Label tidak ditemukan dalam label_map: {e}")
            labels_entry_numeric, labels_exit_numeric = [], []  # Atau lakukan penanganan lain

        features2 = np.array(json.loads(features2))  # bentuk array float

        # Cocokan Kendaraan masuk dan keluar
        is_match = False
        if license_plate_exit == license_plate_entry and vehicle_type_exit == vehicle_type_entry:
            print("Cocok! Data kendaraan sesuai.")
            is_match = True
        else:
            print("Data tidak cocok. Pemeriksaan manual dibutuhkan.")


        # Image Matching logic
        # Gabungkan fitur + label
        try:
            features1 = json.loads(features1_str)
            # features1 = features1.tolist()  # Convert numpy array to list
            # features1 = json.dumps(features1)  # Then convert to JSON stringk
        except json.JSONDecodeError:
            print("[ERROR] Gagal parse JSON dari features1_str")
            features1 = []

        # Pastikan features1_str dan features2 adalah numpy array
        features1_array = np.array(features1, dtype=np.float32)
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
        print(f"Match score: {match_score}")

        # Tentukan status kecocokan
        match_status = "matched" if match_score >= 0.8 else "not_matched"
        print(f"Data {match_status}")

        # Simpan log keluar
        exit_log = VehicleExitLog(
            vehicle_id=vehicle.id,
            vehicle_type=vehicle_type_exit,
            feature=feature,
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
        }), 200


    return app

# MAIN
if __name__ == '__main__':
    app = create_app()
    app.run(debug=False)

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
    app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:admin123@localhost:5432/postgres"
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

    # ROUTES

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

    # Fungsi Pre-Processing untuk OCR
    def preprocess_image(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        binary = cv2.adaptiveThreshold(enhanced, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.GaussianBlur(binary, (5, 5), 0)
        resized = cv2.resize(denoised, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return padded

    # Fungsi Koreksi OCR
    def correct_ocr(text):
        print(text)
        return text.replace("7", "1")

    def detect_license_plate_text(image_file, model_path):
    
        # Konversi file upload ke format OpenCV
        image = convert_uploaded_file_to_cv2(image_file)

        # Simpan ke file temporer agar YOLO bisa baca (YOLOv8 tidak bisa baca NumPy langsung)
        temp_path = "/tmp/temp_uploaded.jpg"
        cv2.imwrite(temp_path, image)

        # Deteksi dengan YOLO
        model = YOLO(model_path)
        results = model(temp_path)

        features = extract_yolo_features(results, model.names)

        # Inisialisasi dulu
        detected_vehicle_label = None
        detected_text = None

        # Tambahkan log untuk label deteksi
        print("Detected objects:")
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                label = model.names[class_id]
                 # Simpan jenis kendaraan (class 0 atau 1)
                if class_id in [0, 1] and detected_vehicle_label is None:
                    detected_vehicle_label = label

                if label.lower() == "plat nomor":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    print(f"Cropping coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    plate_image = image[y1:y2, x1:x2]
                    preprocessed_image = preprocess_image(plate_image)

                    reader = easyocr.Reader(["en", "id"])
                    ocr_results = reader.readtext(preprocessed_image)
                    print(f"OCR raw results: {ocr_results}")

                    for detection in ocr_results:
                        text = detection[1]  # detection[1] adalah string teks yang terbaca
                        confidence = detection[2]             
                        print(f"Detected text: {text}, confidence: {confidence}")

                        if confidence >= 0.55:
                            detected_text = correct_ocr(text)
                            break
                    break

        return detected_vehicle_label, detected_text, features
    
    @app.route('/upload-entry', methods=['POST'])
    def upload_entry():
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image = request.files['image']

        vehicle_type, license_plate, features = detect_license_plate_text(
            image_file=image,
            model_path="yolov8.pt"
        )

        print("License plate text:", license_plate)

        if not image or not license_plate:
            return jsonify({'error': 'Missing data'}), 400

        # buat nama file
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        filename = f"{secure_filename(license_plate)}_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER_IN'], filename)
        feature1 = json.dumps(features.tolist())
        print(features)

        # simpan gambar
        image.save(filepath)

        # simpan ke database
        qr_code_value = f"{license_plate}_{timestamp}"
        vehicle = Vehicle(
            license_plate=license_plate,
            vehicle_type=vehicle_type,
            entry_image_path=filepath,
            qr_code=qr_code_value,
            feature=feature1  # simpan vektor fitur YOLO
            )
        db.session.add(vehicle)
        db.session.commit()

        return jsonify({
            'message': 'Image and data saved successfully',
            'license_plate': license_plate,
            'qr_code': qr_code_value,
            'entry_image_path': filepath
        }), 200


    @app.route('/upload-exit', methods=['POST'])
    def upload_exit():
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image = request.files['image']
        qr_code = request.form.get('qr_code')

        vehicle_type, license_plate, features = detect_license_plate_text(
            image_file=image,
            model_path="yolov8.pt"
        )
        
        vehicle_type_exit = vehicle_type
        license_plate_exit = license_plate
        features2 = features

        if not image or not qr_code:
            return jsonify({'error': 'Missing data'}), 400

        # Cari kendaraan berdasarkan QR code
        vehicle = Vehicle.query.filter_by(qr_code=qr_code).first()
        if not vehicle:
            return jsonify({'error': 'Vehicle not found'}), 404
        
        license_plate_entry = vehicle.license_plate
        vehicle_type_entry = vehicle.vehicle_type
        features1_str = vehicle.feature

        features1 = np.array(json.loads(features1_str))  # bentuk array float

        features2 = features2

        # Cocokan Kendaraan masuk dan keluar
        is_match = False
        if license_plate_exit == license_plate_entry and vehicle_type_exit == vehicle_type_entry:
            print("Cocok! Data kendaraan sesuai.")
            is_match = True
        else:
            print("Data tidak cocok. Pemeriksaan manual dibutuhkan.")

        # Simpan gambar keluar
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        filename = f"{secure_filename(vehicle.license_plate)}_exit_{timestamp}.jpg"
        output_folder = 'static/images/vehicles/keluar'
        os.makedirs(output_folder, exist_ok=True)
        filepath = os.path.join(output_folder, filename)
        image.save(filepath)

        # Image Matching logic
        similarity = cosine_similarity([features1], [features2])[0][0]
        match_score = round(float(similarity), 4)
        match_status = "matched" if similarity >= 0.99 else "not_matched"

        # Simpan log keluar
        exit_log = VehicleExitLog(
            vehicle_id=vehicle.id,
            exit_image_path=filepath,
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
            'exit_image_path': filepath,
            'match_score': match_score,
            'match_status': match_status,
            'vehicle_match': is_match  # Tambahan info cocok/tidak
        }), 200

    return app

# MAIN
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)

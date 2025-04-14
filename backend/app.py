from flask import Flask, jsonify, Response, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from datetime import datetime
from ultralytics import YOLO
import os
import cv2
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image


# Inisialisasi SQLAlchemy dan Marshmallow
db = SQLAlchemy()
ma = Marshmallow()

# MODELS
class Vehicle(db.Model):
    __tablename__ = 'vehicles'
    
    id = db.Column(db.Integer, primary_key=True)
    license_plate = db.Column(db.String(20), nullable=False)
    vehicle_type = db.Column(db.String(50), nullable=True)
    entry_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    entry_image_path = db.Column(db.String(255), nullable=False)
    qr_code = db.Column(db.String(100), nullable=False, unique=True)
    is_exit = db.Column(db.Boolean, nullable=False, default=False)

    exit_logs = db.relationship('VehicleExitLog', backref='vehicle', lazy=True)

class VehicleExitLog(db.Model):
    __tablename__ = 'vehicle_exit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.Integer, db.ForeignKey('vehicles.id'), nullable=False)
    exit_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    exit_image_path = db.Column(db.String(255), nullable=False)
    match_score = db.Column(db.Float, nullable=True)  # 0.0 - 1.0 atau 0 - 100
    match_status = db.Column(db.String(20), nullable=False)  # matched, not_matched, manual_check


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


    @app.route('/upload-entry', methods=['POST'])
    def upload_entry():
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image = request.files['image']
        license_plate = request.form.get('license_plate')
        vehicle_type = request.form.get('vehicle_type', 'mobil')  # default

        if not image or not license_plate:
            return jsonify({'error': 'Missing data'}), 400

        # buat nama file unik
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        filename = f"{secure_filename(license_plate)}_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER_IN'], filename)

        # simpan gambar
        image.save(filepath)

        # simpan ke database
        qr_code_value = f"{license_plate}_{timestamp}"
        vehicle = Vehicle(
            license_plate=license_plate,
            vehicle_type=vehicle_type,
            entry_image_path=filepath,
            qr_code=qr_code_value
        )
        db.session.add(vehicle)
        db.session.commit()

        return jsonify({
            'message': 'Image and data saved successfully',
            'qr_code': qr_code_value,
            'entry_image_path': filepath
        }), 200


    @app.route('/upload-exit', methods=['POST'])
    def upload_exit():
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image = request.files['image']
        qr_code = request.form.get('qr_code')

        print(type(image))
        print(image.content_type)
        print(image.filename)

        if not image or not qr_code:
            return jsonify({'error': 'Missing data'}), 400

        # Cari kendaraan berdasarkan QR code
        vehicle = Vehicle.query.filter_by(qr_code=qr_code).first()
        if not vehicle:
            return jsonify({'error': 'Vehicle not found'}), 404
        
        model = YOLO("yolov8.pt")  # atau model kamu

        results1 = model(vehicle.entry_image_path)
        # Baca sebagai bytes dan konversi ke np.array
        img = Image.open(image.stream).convert('RGB')  # jika pakai PIL
        results2 = model(img)

        features1 = extract_yolo_features(results1, model.names)
        features2 = extract_yolo_features(results2, model.names)

        # Simpan gambar keluar
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        filename = f"{secure_filename(vehicle.license_plate)}_exit_{timestamp}.jpg"
        output_folder = 'static/images/vehicles/keluar'
        os.makedirs(output_folder, exist_ok=True)
        filepath = os.path.join(output_folder, filename)
        image.save(filepath)

        # Matching logic
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

        return jsonify({
            'message': 'Exit data saved successfully',
            'license_plate': vehicle.license_plate,
            'exit_image_path': filepath,
            'match_score': match_score,
            'match_status': match_status
        }), 200
    
    return app

# MAIN
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)

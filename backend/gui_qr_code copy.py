
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import requests
import json
import numpy as np
import threading
import os
from datetime import datetime
import subprocess  # Added for running external scripts

class AdvancedParkingSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Parking System")
        self.root.geometry("1200x800")
        
        # Variabel sistem
        # self.camera_ip = "http://192.168.1.6:4747/video"  # Default webcam
        # self.camera_ip = "http://192.168.137.46:4747/video"  # Default webcam
        self.camera_ip = 0  # Default webcam
        self.cap = None
        self.scan_active = False
        self.qr_data = None
        self.api_url = "http://localhost:5000/get-latest-qr"
        self.save_dir = "captured_images"
        self.model = None  # Akan di-load jika menggunakan model YOLO
        
        # Buat direktori penyimpanan
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Frame utama
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        self.header = ttk.Label(self.main_frame, 
                              text="SMART PARKING SYSTEM",
                              font=('Helvetica', 24, 'bold'))
        self.header.pack(pady=10)
        
        
        # Inisialisasi kamera
        self.cap_left = None
        self.cap_right = None
        self.is_camera_active = False
        
        # Buat frame kamera
        self.create_camera_frames()
        
        # Tombol kontrol
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(self.control_frame, text="Start Cameras", command=self.start_cameras)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(self.control_frame, text="Stop Cameras", command=self.stop_cameras, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        
        # connect btn
        self.connect_btn = ttk.Button(self.control_frame, 
                                    text="Scan QR Code", 
                                    command=self.connect_camera)
        self.connect_btn.grid(row=0, column=2, padx=10)

        self.run_entry_btn = ttk.Button(self.control_frame,
                                      text="Run Entry Camera",
                                      command=self.run_entry_script)
        self.run_entry_btn.grid(row=0, column=4, padx=10)
        
        # Frame API
        self.api_frame = ttk.Frame(self.main_frame)
        self.api_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.api_frame, text="API Endpoint:").grid(row=0, column=0, padx=5)
        self.api_entry = ttk.Entry(self.api_frame, width=50)
        self.api_entry.grid(row=0, column=1, padx=5)
        self.api_entry.insert(0, self.api_url)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to connect camera")
        self.status_bar = ttk.Label(self.main_frame, 
                                  textvariable=self.status_var,
                                  relief=tk.SUNKEN,
                                  anchor=tk.W)
        self.status_bar.pack(fill=tk.X, pady=5)
        
        # Style
        self.style = ttk.Style()
        self.style.configure('Success.TLabelframe', background='lightgreen')
        self.style.configure('Error.TLabelframe', background='lightcoral')
        self.stop_camera = False

        self.run_app_script()
    

    def create_camera_frames(self):
        """Membuat frame untuk kamera kiri dan kanan"""
        # Frame container untuk kedua kamera
        self.cameras_container = ttk.Frame(self.main_frame)
        self.cameras_container.pack(fill=tk.BOTH, expand=True)
        
        # Kamera kiri
        self.left_frame = ttk.LabelFrame(self.cameras_container, text="Camera Left", padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.left_label = ttk.Label(self.left_frame)
        self.left_label.pack(fill=tk.BOTH, expand=True)
        
        # Kamera kanan
        self.right_frame = ttk.LabelFrame(self.cameras_container, text="Camera Right", padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.right_label = ttk.Label(self.right_frame)
        self.right_label.pack(fill=tk.BOTH, expand=True)

    def run_app_script(self):
        try:
            # Run the entry.py script using the system's Python interpreter
            subprocess.Popen(["python", "app.py"])
            self.update_result("Parking system started successfully")
            self.status_var.set("Parking system is running")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run parking system: {str(e)}")
            self.status_var.set("Failed to start parking system")

    def run_entry_script(self):
        try:
            subprocess.Popen(["python", "entry_cam_resnet50.py"])
            self.update_result("Entry system started successfully")
            self.status_var.set("Entry system is running")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run entry system: {str(e)}")
            self.status_var.set("Failed to start entry system")

    
    def connect_camera(self):
        self.cap_left = cv2.VideoCapture(0)
        self.status_var.set(f"Connecting to Camera Entry...")
        
        try:
            if self.cap is not None:
                self.cap.release()
            
            # Coba konversi ke int jika input adalah angka (untuk webcam)
            try:
                camera_source = int(self.camera_ip)
            except ValueError:
                camera_source = self.camera_ip
                
            self.cap = cv2.VideoCapture(camera_source)
            
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
            
            # self.scan_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Connected to {self.camera_ip}")
            
            # Mulai preview kamera
            self.update_camera_preview()
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Error: {str(e)}")
            self.status_var.set("Camera connection failed")
    
    def update_camera_preview(self):
        if self.stop_camera:
            return  # Berhenti jika diminta
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()

            if ret:
                # QR code detection langsung di sini
                detector = cv2.QRCodeDetector()
                data, vertices, _ = detector.detectAndDecode(frame)

                if data and data.strip() and not self.qr_data:
                    self.qr_data = data.strip()

                    # Gambar kotak di QR code
                    if vertices is not None:
                        pts = vertices[0].astype(int)
                        for i in range(len(pts)):
                            cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1) % len(pts)]),
                                    (0, 255, 0), 3)

                    # Tampilkan frame dengan kotak
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img.thumbnail((800, 600))
                    photo = ImageTk.PhotoImage(image=img)
                    self.camera_label.configure(image=photo)
                    self.camera_label.image = photo

                    self.update_result(f"QR Code Detected: {self.qr_data}")
                    self.stop_camera = True
                    self.status_var.set("QR detected - Processing vehicle...")
                    self.run_entry_btn.config(state=tk.NORMAL)
                    # self.run_app_btn.config(state=tk.NORMAL)

                    # Proses kendaraan dalam thread terpisah
                    threading.Thread(target=self.process_and_send, args=(frame_rgb,), daemon=True).start()
                else:
                    # Tampilkan frame biasa jika belum deteksi QR
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img.thumbnail((800, 600))
                    photo = ImageTk.PhotoImage(image=img)
                    self.camera_label.configure(image=photo)
                    self.camera_label.image = photo

            # Lanjutkan preview
            self.root.after(10, self.update_camera_preview)

    def toggle_scan(self):
        if not self.scan_active:
            self.scan_active = True
            # self.scan_btn.config(text="Stop Scan")
            self.status_var.set("Scanning for QR code...")
            self.scan_qr_code()
        else:
            self.scan_active = False
            # self.scan_btn.config(text="Start QR Scan")
            self.status_var.set("Scan stopped")
            self.update_camera_preview()
    
    def scan_qr_code(self):
        if self.cap and self.cap.isOpened() and self.scan_active:
            ret, frame = self.cap.read()
            
            if ret:
                # Deteksi QR code
                detector = cv2.QRCodeDetector()
                data, vertices, _ = detector.detectAndDecode(frame)
                
                if data and data.strip():
                    self.qr_data = data.strip()
                    self.scan_active = False
                    # self.scan_btn.config(text="Start QR Scan")
                    
                    # Gambar bounding box
                    if vertices is not None:
                        pts = vertices[0].astype(int)
                        for i in range(len(pts)):
                            cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1) % len(pts)]), 
                                    (0, 255, 0), 3)
                    
                    # Tampilkan frame dengan QR terdeteksi
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    img.thumbnail((800, 600))
                    photo = ImageTk.PhotoImage(image=img)
                    self.camera_label.configure(image=photo)
                    self.camera_label.image = photo
                    # Hapus gambar dari camera_label
                    self.root.after(self.clear_camera_label)
                    self.camera_label.configure(image=None)
                    self.camera_label.image = None  # Hapus referensi gambar
                    
                    # Update hasil
                    self.update_result(f"QR Code Detected: {self.qr_data}")
                    self.camera_frame.config(style='Success.TLabelframe')
                    self.status_var.set("QR detected - Processing vehicle...")
                    self.run_exit_script.config(state=tk.NORMAL)  # Aktifkan tombol run_entry_script
                    
                    # Proses deteksi kendaraan dan kirim ke API
                    threading.Thread(target=self.process_and_send, args=(frame,), daemon=True).start()
                    return
            
            # Lanjutkan scan jika belum menemukan QR
            if self.scan_active:
                self.root.after(10, self.scan_qr_code)
    
    def process_and_send(self, frame):
        try:
            # Simpan frame asli
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_path = os.path.join(self.save_dir, f"original_{timestamp}.jpg")
            cv2.imwrite(original_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            # self.matched_frame.config(style='Success.TLabelframe')
            
            # Simpan gambar terdeteksi
            annotated_path = os.path.join(self.save_dir, f"annotated_{timestamp}.jpg")
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Vehicle Detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(annotated_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            
            # Siapkan data untuk API
            data = {
                "qr_code": self.qr_data,
            }
            
            # Kirim ke API endpoint pertama
            response = requests.post(self.api_entry.get(), json=data)
            
            if response.status_code == 200:
                self.update_result("Vehicle processed successfully")
                self.update_result(f"API Response: {response.text}")
                result = response.json()
                
                self.update_result("\nVehicle Exit Results:")
                self.show_status_icon("check.png", "green")
                self.update_result(f"License Plate: {result.get('license_plate', 'N/A')}")
                self.update_result(f"Match Score: {result.get('match_score', 'N/A')}")
                
                # Check match status and display appropriate icon
                match_status = result.get('match_status', 'N/A').lower()
                if match_status == 'matched':
                    self.show_status_icon("check.png", "green")
                    self.update_result(f"Match Status: {match_status}")
                    # Ubah tampilan camera_frame
                    self.camera_frame.config(text=" MATCHED - Camera Preview ")  # Ganti judul frame
                    self.camera_frame.configure(style='Green.TLabelframe')  # Ganti style jika ada
                    
                    # Contoh: ubah background label jadi hijau muda
                    self.camera_label.config(background='#e8f5e9')  # Warna hijau muda
                elif match_status == 'not_matched':
                    self.show_status_icon("wrong.png", "red")
                    self.update_result(f"Match Status: {match_status}")
                else:
                    self.update_result(f"Match Status: {match_status}")
                
                self.status_var.set("Vehicle exit processed successfully")
            else:
                self.update_result(f"API Error: {response.text}")
                self.status_var.set("Vehicle processing failed")
                self.camera_frame.config(style='Error.TLabelframe')
            
        except Exception as e:
            self.update_result(f"Error: {str(e)}")
            self.status_var.set("Processing failed")
            self.camera_frame.config(style='Error.TLabelframe')
        
        # Kembali ke mode preview
        self.root.after(0, self.update_camera_preview)

    def show_status_icon(self, icon_file, bg_color):
        """Show a status icon (check or X) in the camera frame"""
        try:
            # Load icon image
            icon_img = Image.open(icon_file)  # Make sure you have these image files
            icon_img.thumbnail((100, 100))
            icon_photo = ImageTk.PhotoImage(icon_img)
            
            # Create or update the status icon label
            if hasattr(self, 'status_icon_label'):
                self.status_icon_label.config(image=icon_photo, background=bg_color)
                self.status_icon_label.image = icon_photo
            else:
                self.status_icon_label = ttk.Label(self.camera_frame, image=icon_photo)
                self.status_icon_label.image = icon_photo
                self.status_icon_label.place(relx=0.9, rely=0.1, anchor=tk.NE)  # Position in top-right
            
            # Remove the icon after 3 seconds
            self.root.after(3000, self.hide_status_icon)
        except Exception as e:
            print(f"Error showing status icon: {str(e)}")

    def hide_status_icon(self):
        """Hide the status icon"""
        if hasattr(self, 'status_icon_label'):
            self.status_icon_label.place_forget()
    
    def update_result(self, message):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)
    
    def on_closing(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedParkingSystem(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()# 
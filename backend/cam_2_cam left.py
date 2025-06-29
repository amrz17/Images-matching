import tkinter as tk
from tkinter import ttk, messagebox, Label, Button
import cv2
from PIL import Image, ImageTk
import threading

from entry_cam import detect_objects_from_frame

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Inspection System")
        
        # Main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self.header = ttk.Label(self.main_frame, 
                              text="SMART PARKING SYSTEM",
                              font=('Helvetica', 24, 'bold'))
        self.header.pack(pady=10)
        
        # Inisialisasi kamera
        self.cap_left = 0
        self.cap_right = 0
        # self.cap_right = "http://192.168.137.46:4747/video"
        self.is_camera_left_active = False
        self.is_camera_right_active = False
        
        # Buat frame kamera
        self.create_camera_frames()

        # Tombol kontrol System
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=10)
        
        self.start_left_btn = ttk.Button(self.control_frame, text="Run Camera 1", command=self.start_camera_left)
        self.start_left_btn.pack(side=tk.LEFT, padx=5)

        self.label_camera_left = tk.Label(self.left_frame)  # atau self.main_frame
        self.label_camera_left.pack()


        # Tombol Deteksi
        self.detect_btn = tk.Button(self.control_frame, text="Deteksi Entry Vehicle", command=self.detect_from_left)
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        
        # Tombol Kamera Kendaraan Keluar
        self.start_right_btn = ttk.Button(self.control_frame, text="Scan QR Code for Exit", command=self.start_camera_right)
        self.start_right_btn.pack(side=tk.LEFT, padx=5)
        
        # Tombol Stop Kamera
        self.stop_btn = ttk.Button(self.control_frame, text="Stop All", command=self.stop_cameras, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Frame hasil
        self.result_frame = ttk.LabelFrame(self.main_frame, 
                                         text=" Detection Results ",
                                         padding=10)
        self.result_frame.pack(fill=tk.BOTH, pady=10)
        
        self.result_text = tk.Text(self.result_frame, 
                                 height=10,
                                 state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Label hasil deteksi
        self.result_label_left = tk.Label(self.left_frame, text="Hasil Deteksi")
        self.result_label_left.pack()
        
        
    def create_camera_frames(self):
        self.cameras_container = ttk.Frame(self.main_frame)
        self.cameras_container.pack(fill=tk.BOTH, expand=True)
        
        # Kamera Kendaraan Masuk
        self.left_frame = ttk.LabelFrame(self.cameras_container, text="Entry Vehicle Camera", padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.left_label = ttk.Label(self.left_frame)
        self.left_label.pack(fill=tk.BOTH, expand=True)
        
        # Kamera Kendaraan Keluar
        self.right_frame = ttk.LabelFrame(self.cameras_container, text="Exit Vehicle Camera", padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.right_label = ttk.Label(self.right_frame)
        self.right_label.pack(fill=tk.BOTH, expand=True)

    def start_camera_left(self):
        if not self.is_camera_left_active:
            self.cap_left = cv2.VideoCapture(0)
            if not self.cap_left.isOpened():
                messagebox.showerror("Error", "Tidak dapat mengakses kamera 1")
                return
            self.is_camera_left_active = True
            self.stop_btn.config(state=tk.NORMAL)
            self.update_frame_left()
    
    def update_frame_left(self):
        if self.is_camera_left_active:
            ret, frame = self.cap_left.read()
            if ret:
                self.current_frame_left = frame.copy()  # Simpan frame terakhir untuk deteksi
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.left_label.configure(image=imgtk)
                self.left_label.imgtk = imgtk  # simpan referensi agar gambar tidak hilang
            self.root.after(10, self.update_frame_left)
    
    def detect_from_left(self):
        if hasattr(self, 'current_frame_left') and self.current_frame_left is not None:
            annotated_frame, labels = detect_objects_from_frame(self.current_frame_left)

            # Tampilkan hasil anotasi di GUI
            rgb_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_annotated)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label_camera_left.imgtk = imgtk  # ✅ BENAR
            self.label_camera_left.configure(image=imgtk)
            self.stop_camera_left(img)

            # (Opsional) tampilkan hasil label
            # label_str = ", ".join(labels)
            # self.result_label_left.config(text=f"Hasil Deteksi: {label_str}")
        else:
            messagebox.showwarning("Peringatan", "Frame belum tersedia")

    def stop_camera_left(self, img):
        if self.cap_left and self.is_camera_left_active:
            self.cap_left.release()
            self.is_camera_left_active = False
            imgtk = ImageTk.PhotoImage(image=img)
            self.left_label.config(image=imgtk)  # Kosongkan tampilan gambar
            # self.label_camera_left.config(image='')  # Kosongkan hasil deteksi (jika perlu)


    def start_camera_right(self):
        if not self.is_camera_right_active:
            self.cap_right = cv2.VideoCapture(0)
            if not self.cap_right.isOpened():
                messagebox.showerror("Error", "Tidak dapat mengakses kamera 2")
                return
            self.is_camera_right_active = True
            self.stop_btn.config(state=tk.NORMAL)
            self.update_frame_right()
    
    def update_frame_right(self):
        if self.is_camera_right_active and self.cap_right:
            ret, frame = self.cap_right.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.right_label.imgtk = imgtk
                self.right_label.config(image=imgtk)
            self.root.after(10, self.update_frame_right)


    def stop_cameras(self):
        if self.cap_left:
            self.cap_left.release()
        if self.cap_right:
            self.cap_right.release()
        
        self.is_camera_left_active = False
        self.is_camera_right_active = False

        self.left_label.config(image='')
        self.right_label.config(image='')

        self.stop_btn.config(state=tk.DISABLED)
    
    def on_close(self):
        self.stop_cameras()
        self.root.destroy()

    def __del__(self):
        self.stop_cameras()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
    root.protocol("WM_DELETE_WINDOW", app.on_close)


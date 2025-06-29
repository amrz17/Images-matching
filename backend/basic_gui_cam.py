import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from camera_module import Camera

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Initialize camera
        self.camera = Camera()
        
        # Create GUI elements
        self.create_widgets()
        
        # Start video update
        self.update()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.mainloop()
    
    def create_widgets(self):
        # Canvas for video display
        self.canvas = tk.Canvas(
            self.window, 
            width=self.camera.width, 
            height=self.camera.height
        )
        self.canvas.pack()
        
        # Button frame
        btn_frame = ttk.Frame(self.window)
        btn_frame.pack(pady=10)
        
        # Snapshot button
        self.btn_snapshot = ttk.Button(
            btn_frame, 
            text="Ambil Foto", 
            command=self.snapshot
        )
        self.btn_snapshot.pack(side=tk.LEFT, padx=5)
        
        # Toggle camera button
        self.btn_toggle = ttk.Button(
            btn_frame, 
            text="Hentikan Kamera", 
            command=self.toggle_camera
        )
        self.btn_toggle.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status = ttk.Label(
            self.window, 
            text="Kamera aktif", 
            foreground="green"
        )
        self.status.pack()
    
    def snapshot(self):
        filename = self.camera.snapshot()
        if filename:
            self.status.config(text=f"Foto disimpan: {filename}", foreground="blue")
        else:
            self.status.config(text="Gagal mengambil foto", foreground="red")
    
    def toggle_camera(self):
        is_running = self.camera.toggle_camera()
        if is_running:
            self.btn_toggle.config(text="Hentikan Kamera")
            self.status.config(text="Kamera aktif", foreground="green")
        else:
            self.btn_toggle.config(text="Mulai Kamera")
            self.status.config(text="Kamera dihentikan", foreground="red")
    
    def update(self):
        ret, frame = self.camera.get_frame()
        
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.window.after(10, self.update)
    
    def on_close(self):
        self.camera.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, "Aplikasi Kamera")
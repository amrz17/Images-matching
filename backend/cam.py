import cv2
from ultralytics import YOLO

# Index camera
cam_index = 0

detection_model = YOLO("yolov8.pt")

def open_camera():
    # Open camera
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"Failed to open camera on index {cam_index}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        results = detection_model.predict(frame, conf=0.5)
        annotated_frame = results[0].plot()

        cv2.imshow("Camera", annotated_frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

open_camera()

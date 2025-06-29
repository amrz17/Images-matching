import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    fgmask = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)[1]

    # Perform object detection
    results = model(frame)

    # Process results
    all_detections = []
    human_hand_detections = []
    other_detections = []

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])
            class_name = model.names[cls]

            # Check if the detection overlaps with the foreground mask
            mask_roi = fgmask[y1:y2, x1:x2]
            if np.mean(mask_roi) > 50:  # Adjust this threshold as needed
                detection = (x1, y1, x2, y2, conf, class_name)
                if class_name in ['person', 'hand']:
                    human_hand_detections.append(detection)
                else:
                    other_detections.append(detection)

    # Sort detections by confidence
    human_hand_detections.sort(key=lambda x: x[4], reverse=True)
    other_detections.sort(key=lambda x: x[4], reverse=True)

    # Select top detections
    top_human_hand = human_hand_detections[:2]
    top_other = other_detections[:5]
    all_detections = top_human_hand + top_other

    # Draw top detections on frame
    for x1, y1, x2, y2, conf, class_name in all_detections:
        color = (0, 255, 0) if class_name in ['person', 'hand'] else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'{class_name} {conf:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Print remaining detections to terminal
    print("Other detections:")
    for detection in other_detections[5:]:
        print(f"{detection[5]}: Confidence {detection[4]:.2f}")

    # Convert frame to image for tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    # Update the label with new image
    lbl_video.img_tk = img_tk
    lbl_video.config(image=img_tk)

    # Schedule the next frame update
    root.after(10, update_frame)
def start_detection():
    global running
    running = True
    update_frame()

def stop_detection():
    global running
    running = False
    cap.release()
    cv2.destroyAllWindows()

# Create the main window
root = tk.Tk()
root.title("Object Detection and Tracking")

# Create a label to display the video feed
lbl_video = Label(root)
lbl_video.pack()

# Create start and stop buttons
btn_start = Button(root, text="Start", command=start_detection)
btn_start.pack(side=tk.LEFT, padx=10)

btn_stop = Button(root, text="Stop", command=stop_detection)
btn_stop.pack(side=tk.RIGHT, padx=10)

# Start the GUI event loop
root.mainloop()
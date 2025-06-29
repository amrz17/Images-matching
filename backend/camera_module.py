# WORK WITH BASIC GUI CAM
import cv2
import time

class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.vid = cv2.VideoCapture(self.camera_index)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", camera_index)
        
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.running = True
    
    def get_frame(self):
        if self.running:
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        return (False, None)
    
    def snapshot(self):
        ret, frame = self.get_frame()
        if ret:
            filename = "capture-" + str(time.time()) + ".jpg"
            cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            return filename
        return None
    
    def toggle_camera(self):
        self.running = not self.running
        return self.running
    
    def release(self):
        if self.vid.isOpened():
            self.vid.release()
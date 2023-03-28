import cv2 
import numpy as np 


class Webcam:
    def __init__(self, cam_num):
        self.cap = cv2.VideoCapture(cam_num, cv2.CAP_DSHOW)
    
    def read(self):
        return self.cap.read()[1]
    
    def show(self, frame):
        cv2.imshow("ExpMaskV1", frame)
    
    def mirror(self, frame):
        return cv2.flip(frame, 1)
    
    def wait(self, delay):
        if cv2.waitKey(delay) == ord("q"):
            self.destroy()
            
    def destroy(self):
        self.cap.release()
        self.destroyAllWindows()
        quit()
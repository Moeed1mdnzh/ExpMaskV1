import cv2 
import argparse
import numpy as np 
from utilities import *


class Main:
    def __init__(self, cam, to_mirror, delay):
        self.cam_num = cam
        self.cam = Webcam(cam) 
        self.page_a = Page_A((640, 380, 88, 108))
        self.page_b = Page_B
        self.LF = LandmarkFinder("models/LandmarkFinder.pt")
        self.MG = MaskGenerator("models/MaskGenerator.pt")
        self.to_mirror = to_mirror 
        self.delay = delay
    
    def begin(self):
        while True:
            frame = self.cam.read()
            if self.to_mirror:
                frame = self.cam.mirror(frame)
            clone = frame.copy()
            pos = self.cam.track_pos()
            if pos < 20:
                pos = 20
                self.cam.set_pos(pos)
            pos = pos / 10
            self.page_a.render(frame, pos)
            self.cam.show(frame)
            capture = self.cam.wait(self.delay)
            if capture is not None:
                face = self.page_a.grab(clone)
                face = cv2.resize(face, (88, 108))
                face_clone = face.copy()
                prep = self.LF.prep(face)
                self.cam.cap.release()
                cv2.destroyAllWindows()
                print("[INFO]: Predicting Landmarks...")
                pred = self.LF.search(prep)
                self.LF.draw(face, pred)
                prep = self.MG.prep(np.array(pred))
                print("[INFO]: Generating Mask...")
                pred = self.MG.generate(prep)
                self.page_b = Page_B
                self.page_b =  self.page_b(clone)
                self.page_b.render([face_clone, 
                                    face,
                                    pred])
                capture = None
                cv2.destroyAllWindows()
                self.cam = Webcam(self.cam_num)
                
                
                
    
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--camera", type=int, default=0, help="The particular camera you'd like to use.For instance, 0 means the first camera.")
    ap.add_argument("-m", "--mirror", type=bool, default=False, help="Wether to flip the frames or not.")
    ap.add_argument("-d", "--delay", type=int, default=1, help="The delay for capturing the frames.")
    args = vars(ap.parse_args())
    Main(args["camera"], args["mirror"], args["delay"]).begin()
    
    
if __name__ == "__main__":
    main()
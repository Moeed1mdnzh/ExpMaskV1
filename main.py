import os
import cv2
import argparse
import numpy as np
import mediapipe as mp
from utilities import *


class Main:
    def __init__(self, cam, to_mirror, delay):
        self.cam_num = cam
        self.cam = Webcam(cam)
        self.page_a = Page_A((640, 380, 88, 108))
        self.page_b = Page_B
        self.LF = LandmarkFinder(mp.solutions.face_detection,
                                 mp.solutions.face_mesh)
        self.MG = MaskGenerator(os.sep.join(["model", "MaskGenerator.pt"]))
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
                self.cam.cap.release()
                cv2.destroyAllWindows()
                print("[INFO]: Predicting Landmarks...")
                pred = self.LF.search(face)
                self.LF.draw(face, pred)
                lms = []
                for i in range(0, 10, 2):
                    pts = pred[i: i+2]
                    pts[0] = np.interp(pts[0], (0, 88), (0, 1))
                    pts[1] = np.interp(pts[1], (0, 108), (0, 1))
                    lms.append(pts[0])
                    lms.append(pts[1])
                prep = self.MG.prep(np.array(lms))
                print("[INFO]: Generating Mask...")
                pred = self.MG.generate(prep)
                self.page_b = Page_B
                self.page_b = self.page_b(clone)
                self.page_b.render([face_clone,
                                    face,
                                    pred])
                capture = None
                cv2.destroyAllWindows()
                self.cam = Webcam(self.cam_num)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--camera", type=int, default=0,
                    help="The particular camera you'd like to use.For instance, 0 means the first camera.")
    ap.add_argument("-m", "--mirror", type=bool, default=False,
                    help="Wether to flip the frames or not.")
    ap.add_argument("-d", "--delay", type=int, default=1,
                    help="The delay for capturing the frames.")
    args = vars(ap.parse_args())
    Main(args["camera"], args["mirror"], args["delay"]).begin()


if __name__ == "__main__":
    main()

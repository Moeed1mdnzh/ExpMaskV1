import cv2 
import argparse
import numpy as np 
from utilities import *


class Main:
    def __init__(self, cam, to_mirror, delay):
        self.cam = Webcam(cam) 
        self.to_mirror = to_mirror 
        self.delay = delay
    
    def begin(self):
        while True:
            frame = self.cam.read()
            if self.to_mirror:
                frame = self.cam.mirror(frame)
            self.cam.show(frame)
            self.cam.wait(self.delay)
    
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--camera", type=int, default=0, help="The particular camera you'd like to use.For instance, 0 means the first camera.")
    ap.add_argument("-m", "--mirror", type=bool, default=False, help="Wether to flip the frames or not.")
    ap.add_argument("-d", "--delay", type=int, default=1, help="The delay for capturing the frames.")
    args = vars(ap.parse_args())
    Main(args["camera"], args["mirror"], args["delay"]).begin()
    
    
if __name__ == "__main__":
    main()
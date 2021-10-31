import cv2
import numpy as np
import sys

def main():

    face_detector = cv2.CascadeClassifier("data/haarcascade_frontalfacce_defualt.xml")
    camera = cv2.VideoCapture(0)

    while True:
        got_image, bgr_image = camera.read()
        if not got_image:
            sys.exit()


        cv2.imshow("camera feed", bgr_image)
        cv2.waitKey(0)

    print("all done")


if __name__ == "__main__":
    main()
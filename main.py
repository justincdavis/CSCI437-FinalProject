import cv2
import numpy as np
import sys

def main():

    face_detector = cv2.CascadeClassifier("data/haarcascade_frontalfacce_default.xml")
    camera = cv2.VideoCapture(0)

    while True:
        got_image, bgr_image = camera.read()
        if not got_image:
            sys.exit()


        cv2.imshow("camera feed", bgr_image)
        key_pressed = cv2.waitKey(10) & 0xFF
        if key_pressed == 27:
            break  # Quit on ESC

    print("all done")


if __name__ == "__main__":
    main()

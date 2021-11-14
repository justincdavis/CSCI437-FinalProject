import cv2
import numpy as np
import sys

def main():

    face_detector = cv2.CascadeClassifier("data/haarcascade_frontalfacce_default.xml")
    eye_cascade = cv2.CascadeClassifier("data/haarcascade_eye.xml")
    camera = cv2.VideoCapture(0)

    while True:
        got_image, bgr_image = camera.read()
        if not got_image:
            sys.exit()

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

        cv2.imshow("camera feed", bgr_image)
        key_pressed = cv2.waitKey(10) & 0xFF
        if key_pressed == 27:
            break  # Quit on ESC

    print("all done")


if __name__ == "__main__":
    main()

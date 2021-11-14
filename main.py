import cv2
import numpy as np
import sys
from PIL import Image
from animation import Character, init_Character

def main():

    face_detector = cv2.CascadeClassifier("data/haarcascade_frontalfacce_default.xml")
    eye_cascade = cv2.CascadeClassifier("data/haarcascade_eye.xml")
    camera = cv2.VideoCapture(0)

    #create variables for character and drawing the character
    c  = Character(init_Character())
    scale = 10
    attributes = [0, 0, 0, 0, 0, 0]

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
        
        #identify faces

        #fit bounding boxes

        #choose sprties
            #edit the attrributes list

        surface = Image.new('RGBA', (56, 80))
        c.draw(surface, attributes)
        surface = surface.resize((56*scale, 80*scale), resample=Image.BOX)
        numpy_image=np.array(surface)  
        opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_BGRA2RGBA) 
        cv2.imwrite('output.png', opencv_image) 
        cv2.imshow('output.png', opencv_image)
        key_pressed = cv2.waitKey(10) & 0xFF
        if key_pressed == 27:
            break  # Quit on ESC

    print("all done")


if __name__ == "__main__":
    main() 

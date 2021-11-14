import cv2
import numpy as np
import sys
from PIL import Image
from animation import Character, init_Character, generate_frame

def detectEyes(image, cascade):
    gray_image = cv2.cvtColor(image, cv2.BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_image)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return image

def main():

    face_detector = cv2.CascadeClassifier("data/haarcascade_frontalfacce_default.xml")
    camera = cv2.VideoCapture(0)

    #create variables for character and drawing the character
    c  = Character(init_Character())
    scale = 10
    attributes = [0, 0, 0, 0, 0, 0]
    images = []

    while True:
        got_image, bgr_image = camera.read()
        if not got_image:
            sys.exit()


        cv2.imshow("camera feed", bgr_image)
        key_pressed = cv2.waitKey(10) & 0xFF
        if key_pressed == 27:
            break  # Quit on ESC
        
        #identify faces

        #fit bounding boxes

        #choose sprties
            #edit the attrributes list

        image = generate_frame(c, scale, attributes, images)
        cv2.imshow("output.png", image)
        key_pressed = cv2.waitKey(10) & 0xFF
        if key_pressed == 27:
            break  # Quit on ESC

    #after exiting while loop, read images in array and convert to video

    print("all done")


if __name__ == "__main__":
    main() 

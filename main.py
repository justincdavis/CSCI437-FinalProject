import cv2
import numpy as np
import sys, os
from PIL import Image
from animation import Character, init_Character

def main():

    path = os.path.join(sys.path[0], "data/haarcascade_frontalface_default.xml")
    face_detector = cv2.CascadeClassifier(path)
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
        faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
        print(faces, "\n")
        if faces == ():
            faces = old_faces

        x_face = faces[0][0]
        y_face = faces[0][1]
        face_width = faces[0][2]
        face_height = faces[0][3]

        #face_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        #face_image = cv2.rectangle(face_image, (x_face, y_face), (x_face+face_width, y_face+face_height), (0,0,255), 5)
        face_image = gray_image[int(y_face):int(y_face+face_height), int(x_face):int(x_face+face_width)]

        

        cv2.imshow("camera feed", bgr_image)
        cv2.imshow("test", face_image)
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
        old_faces = faces
        key_pressed = cv2.waitKey(10) & 0xFF
        if key_pressed == 27:
            break  # Quit on ESC

    print("all done")


if __name__ == "__main__":
    main() 

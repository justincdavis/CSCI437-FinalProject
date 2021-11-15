import cv2
import numpy as np
import sys, os
from PIL import Image
from animation import Character, init_Character, generate_frame

# Takes a bgr opencv image, the eye detector cascade, and an optional draw
# If draw is false or not given, returns the eyes given by the cascade and the input image
# If draw is true then the image has the eye bounding boxes drawn on it
def detectEyes(image, cascade, draw=False):
    gray_image = cv2.cvtColor(image, cv2.BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_image)
    if(draw):
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return eyes, image

def crop2Face(image, cascade, last_face):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_image)
    if faces is not None:
        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]
        face_image = gray_image[y:y+h, x:x+w]
        return face_image, faces
    else:
        return None, None

    #resize image
    scale_percent = 220  # percent of original size
    width = int(face_image.shape[1] * scale_percent / 100)
    height = int(face_image.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(face_iamge, dim, interpolation=cv2.INTER_AREA)
    return resized, face

def main():

    path = os.path.join(sys.path[0], "data/haarcascade_frontalface_default.xml")
    face_detector = cv2.CascadeClassifier(path)
    eye_detector = cv2.CascadeClassifier("data/haarcascade_eye.xml")
    camera = cv2.VideoCapture(0)

    #create variables for character and drawing the character
    c  = Character(init_Character())
    scale = 10
    # max values: 0, 13,36,7, 0, 11
    attributes = [0, 0, 0, 0, 0, 0]
    images = []

    last_face = None
    while True:
        got_image, bgr_image = camera.read()
        if not got_image:
            sys.exit()

        face_image, face = crop2Face(bgr_image, face_detector, last_face)
        if face_image is not None and face is not None:
            cv2.imshow("test", face_image)
            cv2.imshow("camera feed", bgr_image)
            last_face = face
            key_pressed = cv2.waitKey(10) & 0xFF
            if key_pressed == 27:
                break  # Quit on ESC

            # identify faces

            # fit bounding boxes

            # choose sprties
            # edit the attrributes list

            image = generate_frame(c, scale, attributes, images)
            cv2.imshow("output.png", image)
            key_pressed = cv2.waitKey(10) & 0xFF
            if key_pressed == 27:
                break  # Quit on ESC



    #after exiting while loop, read images in array and convert to video

    print("all done")


if __name__ == "__main__":
    main() 

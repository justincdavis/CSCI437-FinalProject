import cv2
import numpy as np
import sys, os
from PIL import Image
from animation import Character, init_Character, generate_frame

# Takes a bgr opencv image, the eye detector cascade, and an optional draw
# If draw is false or not given, returns the eyes given by the cascade and the input image
# If draw is true then the image has the eye bounding boxes drawn on it
def detectEyes(image, cascade, draw=False):
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(image)
    if(draw):
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return eyes, image

def crop2Face(image, cascade):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_image)
    if faces is not ():
        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]
        face_image = gray_image[y:y+h, x:x+w]
        return face_image, faces
    else:
        return None, None

def crop2Eyes(image, eyes):
    height, width = image.shape
    if eyes is not ():
        left_eye, right_eye = np.zeros_like(eyes[0]), np.zeros_like(eyes[0])
        i = 0
        for eye in eyes:
            x = eye[0]
            y = eye[1]
            w = eye[2]
            h = eye[3]
            if i < 2:
                if x > width/2:
                    right_eye = image[y:y + h, x:x + w]
                if x <= width/2:
                    left_eye = image[y:y + h, x:x + w]
            i += 1
        if np.sum(right_eye) == 0 or np.sum(left_eye) == 0:
            right_eye, left_eye = image, image
            print("Error detecting eyes. \n")
        return left_eye, right_eye
    else:
        return None, None

def resizeFaceImage(face_image, scale_percent):
    width = int(face_image.shape[1] * scale_percent / 100)
    height = int(face_image.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(face_image, dim, interpolation=cv2.INTER_AREA)
    return resized

def main():

    #Initialize paths for haarcascade detection
    face_path = os.path.join(sys.path[0], "data/haarcascade_frontalface_default.xml")
    face_detector = cv2.CascadeClassifier(face_path)
    eye_path = os.path.join(sys.path[0], "data/haarcascade_eye.xml")
    eye_detector = cv2.CascadeClassifier(eye_path)
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
        #image_height, image_width, _ = bgr_image.shape
        #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        #videoWriter = cv2.VideoWriter("output.avi", fourcc=fourcc, fps=20.0,
                                        #frameSize=(image_width, image_height))
        face_image, face = crop2Face(bgr_image, face_detector)
        if face_image is not None and face is not None:
            #videoWriter.write(bgr_image)
            eyes, eye_image = detectEyes(face_image, eye_detector, draw=True)
            right_eye, left_eye = crop2Eyes(eye_image, eyes)
            if right_eye is not None and left_eye is not None:
                image = generate_frame(c, scale, attributes, images)

                cv2.imshow("output.png", image)
                cv2.imshow("camera feed", bgr_image)
                cv2.imshow("testR", right_eye)
                cv2.imshow("testL", left_eye)

            key_pressed = cv2.waitKey(10) & 0xFF
            if key_pressed == 27:
                break  # Quit on ESC

    print("all done")
if __name__ == "__main__":
    main() 

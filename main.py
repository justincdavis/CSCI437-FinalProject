import cv2
import numpy as np
import sys, os, time
from PIL import Image
from animation import Character, init_Character, get_pupil_pos, generate_frame

# Takes a bgr opencv image, the eye detector cascade, and an optional draw
# If draw is false or not given, returns the eyes given by the cascade and the input image
# If draw is true then the image has the eye bounding boxes drawn on it
def detectEyes(image, cascade, draw=False):
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = image
    eyes = cascade.detectMultiScale(gray_image)
    if(draw):
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return eyes, image


# takes a bgrImage, a morphology kernel, and an optional parameter run
# Transforms the image to binary using otsu method
def binaryMorphology(eyeImage, kernel, run=True):
    if(run):
        gray_image = cv2.cvtColor(eyeImage, cv2.COLOR_BGR2GRAY)
        _, binaryImg = cv2.threshold(gray_image, 48, 255, cv2.THRESH_BINARY)
        filtered_img = cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE, kernel)
        filtered_img = cv2.morphologyEx(filtered_img, cv2.MORPH_OPEN, kernel*2)
        
        return filtered_img
    return eyeImage

# runs cv2.connectedComponentsWithStats
# then inverts binary image and runs again
# returns the actual components for normal and inverted
def getConComps(eyeImage, connected=8):
    # connected components for white blobs
    out = cv2.connectedComponentsWithStats(eyeImage, connected)
    # connected components for former black blobs
    inverted_img = cv2.bitwise_not(eyeImage)
    i_out = cv2.connectedComponentsWithStats(inverted_img, connected)
    return out, i_out

# takes the connected components for normal and inverted image, with optional parameter threshold
def findTargets(whiteBlobs, blackBlobs, threshold=2.0):
    target = None
    maxArea = -1
    i = 0
    num, labs, stats, cents = whiteBlobs
    for lab in labs:
        if stats[lab][0][4] > maxArea:
            maxArea = stats[lab][0][4]
            if i > num:
                i = num-1
            target = cents[i]
        i = i + 1
       
    return target


# determines if a given point (x,y) is within the topleft, topright, botleft, botright quadrant of the image
# top left is 0, top right is 1, bottom left is 2, bottom right is 3
def determineQuadrant(image, point, centerWidth=7):
    x = point[0]
    y = point[1]
    dim = image.shape
    widthBound = (dim[0] - centerWidth) / 2
    widthBound2 = widthBound + centerWidth
    heightBound = (dim[1] - centerWidth) / 2
    heightBound2 = widthBound + centerWidth
    vertPose = 1
    horzPose = 1
    if x < widthBound:
        horzPose = 0
    elif x > widthBound2:
        horzPose = 2
    if y < heightBound:
        vertPose = 0
    elif y > heightBound2:
        vertPose = 2
    return horzPose + 3 * vertPose

def classifyPupils(leftEye, rightEye, debug=False):
    try:
        # create a nxn square box filter
        n = 2
        kernel = np.ones((n, n), np.uint8)
        # compute the binary images and perform morphology, run is optional parameter. Skip morphology if run = False
        binaryLeft = binaryMorphology(leftEye, kernel, run=True)
        binaryRight = binaryMorphology(rightEye, kernel, run=True)

        if debug: # show the binary images if debug mode is on
            cv2.imshow("binaryLeft", binaryLeft)
            cv2.imshow("binaryRight", binaryRight)

        # compute where the connected components are
        leftWhite, leftBlack = getConComps(binaryLeft, connected=8)
        rightWhite, rightBlack = getConComps(binaryRight, connected=8)

        # compute where two points match within a given threshold and return them as a possible target
        possibleLeftTargets, minLeft = findPossibleTargets(leftWhite, leftBlack, threshold=2.0)
        possibleRightTargets, minRight = findPossibleTargets(rightWhite, rightBlack, threshold=2.0)

        # get the smallest distance as our best option for the pupil
        bestLeftTarget = possibleLeftTargets[minLeft]
        bestRightTarget = possibleRightTargets[minRight]

        #determines the quadrant of each pupil
        centerWidth = 15
        leftQuadrant = determineQuadrant(leftEye, bestLeftTarget, centerWidth=centerWidth)
        rightQuadrant = determineQuadrant(rightEye, bestRightTarget, centerWidth=centerWidth)

        if debug:
            print("Quadrants -> left: {}, right: {}".format(leftQuadrant, rightQuadrant))

        return (leftQuadrant, rightQuadrant)
    except:
        return (4, 4)

# takes image and face classifier, returns image cropped to first face and it's relational properties
def crop2Face(image, cascade):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_image)
    if faces is not ():
        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]
        face_image = image[y:y+h, x:x+w]
        return face_image, faces
    else:
        return None, None

# takes face image and eye data from classifier (output of detectEyes()), returns best guess image of left eye and right eye
def crop2Eyes(image, eyes):
    height, width, _ = image.shape
    if eyes is not ():
        left_eye, right_eye = np.zeros_like(image), np.zeros_like(image)
        i = 0
        for eye in eyes:
            x = eye[0]
            y = eye[1]
            w = eye[2]
            h = eye[3]
            if y < 2*height/5:
                if np.sum(right_eye) == 0 and x > width/2:
                    right_eye = image[y + int(h*0.2): y + h, x:x + w]
                if np.sum(left_eye) == 0 and x <= width/2:
                    left_eye = image[y + int(h*0.2): y + h, x:x + w]

        if np.sum(right_eye) == 0:
            right_eye = image
            print("No right eye detected!")
        if np.sum(left_eye) == 0:
            left_eye = image
            print("No left eye detected!")
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

    # camera to videocapture from webcam
    camera = cv2.VideoCapture(0)    
    got_image, bgr_image = camera.read()
    use_delay = False
    if not got_image:
        camera = cv2.VideoCapture("output.avi")
        use_delay = True

    image_height, image_width, _ = bgr_image.shape

    # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # video_path = os.path.join(sys.path[0], "eye.avi")
    # videoWriter = cv2.VideoWriter(video_path, fourcc=fourcc, fps=10.0,
    #                               frameSize=(image_width, image_height))

    # create character object and variables for drawing the character
    c = Character(init_Character())
    scale = 10 # scale of character image (and encapsulating window) drawn 
    # max values: 0, 13,36,7, 0, 11
    attributes = [0, 0, 0, 0, 0, 0]
    images = []

    last_face = None
    while True:
        got_image, bgr_image = camera.read()
        if not got_image:
            sys.exit()

        cv2.imshow("camera feed", bgr_image)

        face_image, faces = crop2Face(bgr_image, face_detector)
        if face_image is not None and faces is not None:
            # videoWriter.write(bgr_image)
            cv2.imshow("face image", face_image)
            gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            eyes, eye_image = detectEyes(gray_image, eye_detector, draw=False)
            right_eye, left_eye = crop2Eyes(face_image, eyes)

            try:
                right_eye = cv2.resize(right_eye, (image_width, image_height), interpolation= cv2.INTER_LINEAR)
                left_eye = cv2.resize(left_eye, (image_width, image_height), interpolation= cv2.INTER_LINEAR)
            except Exception:
                print(Exception)
                continue
    
            cv2.imshow("left eye", left_eye)
            cv2.imshow("right eye", right_eye)

            if right_eye is not None and left_eye is not None:                
               
                n = 8
                kernel = np.ones((n, n), np.uint8)
                # compute the binary images and perform morphology, run is optional parameter. Skip morphology if run = False
                binaryRight = binaryMorphology(right_eye, kernel, run=True)
                binaryLeft = binaryMorphology(left_eye, kernel, run=True)
               
                cv2.imshow("morph left eye", binaryLeft)
                cv2.imshow("morph right eye", binaryRight)

                # compute where the connected components are
                rightWhite, rightBlack = getConComps(binaryRight, connected=8)
                leftWhite, leftBlack = getConComps(binaryLeft, connected=8)

                # compute where two points match within a given threshold and return them as a possible target
                possibleRightTarget = findTargets(rightWhite, rightBlack, threshold=2.0)
                possibleLeftTarget = findTargets(leftWhite, leftBlack, threshold=2.0)

                if possibleRightTarget is not None and possibleRightTarget.any != np.NaN:
                    cv2.circle(right_eye, (int(possibleRightTarget[0]), int(possibleRightTarget[1])), 20, (0, 0, 255, 255), 1)
                if possibleLeftTarget is not None and possibleLeftTarget.any != np.NaN:
                    cv2.circle(left_eye, (int(possibleLeftTarget[0]), int(possibleLeftTarget[1])), 20, (0, 0, 255, 255), 1)

                # cv2.putText(right_eye, str(eyeQuads[0]), (20, 55), color=(0, 0, 255, 255),
                #             fontFace=1,  fontScale=5.5, thickness=2)

                cv2.line(right_eye, (0, int(image_height*11/32)), (int(image_width), int(image_height*11/32)), (0, 0, 255, 255), 1)
               
                cv2.line(right_eye, (int(image_width*15/32), int(0)), (int(image_width*15/32), int(image_height)), (0, 0, 255, 255), 1)
                cv2.line(right_eye, (int(image_width*9/16), int(0)), (int(image_width*9/16), int(image_height)), (0, 0, 255, 255), 1)


                cv2.line(left_eye, (0, int(image_height*11/32)), (int(image_width), int(image_height*11/32)), (0, 0, 255, 255), 1)
               
                cv2.line(left_eye, (int(image_width*17/32), int(0)), (int(image_width*17/32), int(image_height)), (0, 0, 255, 255), 1)
                cv2.line(left_eye, (int(image_width*7/16), int(0)), (int(image_width*7/16), int(image_height)), (0, 0, 255, 255), 1)

                eye_sect = 4
                if (possibleRightTarget[1] < image_height*11/32):
                    if (possibleRightTarget[0] < image_width*15/32):
                        eye_sect = 0
                    elif (possibleRightTarget[0] > image_width*9/16):
                        eye_sect = 2
                    else:
                        eye_sect = 1
                else:
                    if (possibleRightTarget[0] < image_width*15/32):
                        eye_sect = 3
                    elif (possibleRightTarget[0] > image_width*9/16):
                        eye_sect = 5
                    else:
                        eye_sect = 4

                attributes[2] = get_pupil_pos(eye_sect)
                image = generate_frame(c, scale, attributes, images)

                # videoWriter.write(right_eye)

                cv2.imshow("output.png", image)
                
                cv2.imshow("testR", right_eye)
                cv2.imshow("testL", left_eye)


            key_pressed = cv2.waitKey(10) & 0xFF
            if key_pressed == 27:
                break  # Quit on ESC
            if use_delay:
                time.sleep(.1)


    #after exiting while loop, read images in array and convert to video

    print("all done")
    # videoWriter.release()


if __name__ == "__main__":
    main()

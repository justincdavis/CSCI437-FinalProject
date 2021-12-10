import cv2
import numpy as np
import sys, os, time
from PIL import Image
from animation import Character, init_Character, get_pupil_pos, generate_frame

# Takes a bgr opencv image, the eye detector cascade, and an optional draw
# If draw is false or not given, returns the eyes given by the cascade and the input image
# If draw is true then the image has the eye bounding boxes drawn on it
def detectEyes(image, cascade, draw=False):
    gray_image = image
    eyes = cascade.detectMultiScale(gray_image)
    if(draw):
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return eyes, image


# takes a bgrImage, a morphology kernel, and an optional parameter run
# Transforms the image to binary using cvtColor, thresholding, and closing/opening morphologies
def binaryMorphology(eyeImage, kernel, run=True):
    if(run):
        gray_image = cv2.cvtColor(eyeImage, cv2.COLOR_BGR2GRAY)
        _, binaryImg = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)
        filtered_img = cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE, kernel)
        filtered_img = cv2.morphologyEx(filtered_img, cv2.MORPH_OPEN, kernel*2)
        return filtered_img
    return eyeImage

# takes the connected components for binary image, with optional parameter threshold
# returns blob centroid with largest area
def findTargets(blackBlobs, threshold=2.0):
    target = None
    maxArea = -1
    i = 0
    num, labs, stats, cents = blackBlobs
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

# takes in targets and returrns the region of the pupil
def classifyPupil(possibleRightTarget, possibleLeftTarget, right_eye, left_eye, last_eye, numLeft, numRight):
    image_height, image_width, _ = left_eye.shape
    # simply draw circles on the possible detected pupils
    # TODO: remove issue of findTargets returning NaN instead of None
    if possibleRightTarget is not None and possibleRightTarget.any != np.NaN:
        try:
            cv2.circle(right_eye, (int(possibleRightTarget[0]), int(possibleRightTarget[1])), 20, (0, 0, 255, 255), 1)
        except Exception:
            None
    if possibleLeftTarget is not None and possibleLeftTarget.any != np.NaN:
        try:
            cv2.circle(left_eye, (int(possibleLeftTarget[0]), int(possibleLeftTarget[1])), 20, (0, 0, 255, 255), 1)
        except Exception:
            None

    # draw funny little lines on right eye (for classification regions)
    # TODO: integrate into function with actual region definition
    cv2.line(right_eye, (0, int(image_height*13/32)), (int(image_width), int(image_height*13/32)), (0, 0, 255, 255), 1)

    cv2.line(right_eye, (int(image_width*15/32), int(0)), (int(image_width*15/32), int(image_height)), (0, 0, 255, 255), 1)
    cv2.line(right_eye, (int(image_width*10/16), int(0)), (int(image_width*10/16), int(image_height)), (0, 0, 255, 255), 1)

    # draw funny little lines on left eye (for classification regions)
    cv2.line(left_eye, (0, int(image_height*13/32)), (int(image_width), int(image_height*13/32)), (0, 0, 255, 255), 1)

    cv2.line(left_eye, (int(image_width*15/32), int(0)), (int(image_width*15/32), int(image_height)), (0, 0, 255, 255), 1)
    cv2.line(left_eye, (int(image_width*10/16), int(0)), (int(image_width*10/16), int(image_height)), (0, 0, 255, 255), 1)

    # classification region
    # TODO: integrate into function with drawn lines
    if numRight < numLeft and numRight != 0:
        bestTarget = possibleRightTarget
    elif numLeft != 0:
        bestTarget = possibleLeftTarget
    else:
        bestTarget = possibleLeftTarget

    eye_sect = last_eye
    if (bestTarget[1] < image_height*13/32):
        if (bestTarget[0] < image_width*15/32):
            eye_sect = 0
        elif (bestTarget[0] > image_width*10/16):
            eye_sect = 2
        else:
            eye_sect = 1
    else:
        if (bestTarget[0] < image_width*15/32):
            eye_sect = 3
        elif (bestTarget[0] > image_width*10/16):
            eye_sect = 5
        else:
            eye_sect = 4

    return eye_sect, right_eye, left_eye

# takes image and face classifier,
# returns image cropped to first face and it's relational properties
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

# takes face image and eye data from classifier (output of detectEyes()),
# returns best guess image of left eye and right eye
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
            # only consider eyes in the upper part of the image
            if y < 2*height/5:
                if np.sum(right_eye) == 0 and x > width/2:
                    right_eye = image[y + int(h*0.2): y + h, x:x + w]
                if np.sum(left_eye) == 0 and x <= width/2:
                    left_eye = image[y + int(h*0.2): y + h, x:x + w]

        if np.sum(right_eye) == 0:
            right_eye = None
            print("No right eye detected!")
        if np.sum(left_eye) == 0:
            left_eye = None
            print("No left eye detected!")
        return left_eye, right_eye
    else:
        return None, None


def main():
    # Initialize paths for haarcascade detection
    face_path = os.path.join(sys.path[0], "data/haarcascade_frontalface_default.xml")
    face_detector = cv2.CascadeClassifier(face_path)
    eye_path = os.path.join(sys.path[0], "data/haarcascade_eye.xml")
    eye_detector = cv2.CascadeClassifier(eye_path)

    # camera to videocapture from webcam
    camera = cv2.VideoCapture(1)
    got_image, bgr_image = camera.read()
    use_delay = False
    # if no webcam detected (also maybe add command line option) then use video input instead
    if not got_image:
        camera = cv2.VideoCapture("output.avi")
        use_delay = True

    got_image, bgr_image = camera.read()
    image_height, image_width, _ = bgr_image.shape

    # for creating images for demo
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video_path = os.path.join(sys.path[0], "face.avi")
    face_videoWriter = cv2.VideoWriter(video_path, fourcc=fourcc, fps=10.0,
                                  frameSize=(image_width, image_height))

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video_path = os.path.join(sys.path[0], "reye.avi")
    reye_videoWriter = cv2.VideoWriter(video_path, fourcc=fourcc, fps=10.0,
                                  frameSize=(image_width, image_height))

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video_path = os.path.join(sys.path[0], "leye.avi")
    leye_videoWriter = cv2.VideoWriter(video_path, fourcc=fourcc, fps=10.0,
                                  frameSize=(image_width, image_height))

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video_path = os.path.join(sys.path[0], "breye.avi")
    breye_videoWriter = cv2.VideoWriter(video_path, fourcc=fourcc, fps=10.0,
                                  frameSize=(image_width, image_height))

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video_path = os.path.join(sys.path[0], "bleye.avi")
    bleye_videoWriter = cv2.VideoWriter(video_path, fourcc=fourcc, fps=10.0,
                                  frameSize=(image_width, image_height))

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video_path = os.path.join(sys.path[0], "character.avi")
    character_videoWriter = cv2.VideoWriter(video_path, fourcc=fourcc, fps=10.0,
                                  frameSize=(int(image_width/2), image_height))

    # create character object and variables for drawing the character
    c = Character(init_Character())
    scale = 10 # scale of character image (and encapsulating window) drawn 
    # max values: 0, 13,36,7, 0, 11
    attributes = [0, 0, 0, 0, 0, 0]
    images = []

    last_face = None
    last_eye = 4
    while True:
        got_image, bgr_image = camera.read()
        if not got_image:
            sys.exit()

        # cv2.imshow("camera feed", bgr_image) # window for camera feed
        
        face_image, faces = crop2Face(bgr_image, face_detector)
        if face_image is not None and faces is not None:
            # cv2.imshow("face image", face_image) # window for cropped face
            gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            eyes, eye_image = detectEyes(gray_image, eye_detector, draw=False)
            right_eye, left_eye = crop2Eyes(face_image, eyes)

            try:
                right_eye = cv2.resize(right_eye, (image_width, image_height), interpolation= cv2.INTER_CUBIC)
                left_eye = cv2.resize(left_eye, (image_width, image_height), interpolation= cv2.INTER_CUBIC)
            except Exception:
                print(Exception)
                print("could not upscale eye images")
                continue
    
            # # windows for eyes
            # cv2.imshow("left eye", left_eye) 
            # cv2.imshow("right eye", right_eye)

            # if both eyes are detected
            # TODO: expand options for single eye use if one is obstructed
            if right_eye is not None and left_eye is not None:     
                # for creating demo videos  
                face_image = cv2.resize(face_image, (image_width, image_height), interpolation= cv2.INTER_CUBIC) # upscale face image
                face_videoWriter.write(face_image)

                n = 8
                kernel = np.ones((n, n), np.uint8)
                # compute the binary images and perform morphology
                binaryRight = binaryMorphology(right_eye, kernel)
                binaryLeft = binaryMorphology(left_eye, kernel)
               
                cv2.imshow("morph left eye", binaryLeft) # window for binary eye
                cv2.imshow("morph right eye", binaryRight) # window for binary eye
                bleye_img = cv2.cvtColor(binaryLeft, cv2.COLOR_GRAY2BGR)
                bleye_videoWriter.write(bleye_img) # for creating demo videos  
                breye_img = cv2.cvtColor(binaryRight, cv2.COLOR_GRAY2BGR)
                breye_videoWriter.write(breye_img) # for creating demo videos  

                # compute where the connected components are
                rightWhite = cv2.connectedComponentsWithStats(binaryRight, connectivity=8)
                leftWhite = cv2.connectedComponentsWithStats(binaryLeft, connectivity=8)
                numLeft, numRight = rightWhite[0], leftWhite[0]

                # compute where two points match within a given threshold and return them as a possible target
                possibleRightTarget = findTargets(rightWhite, threshold=2.0)
                possibleLeftTarget = findTargets(leftWhite, threshold=2.0)


                eye_sect, right_eye, left_eye = classifyPupil(possibleRightTarget, possibleLeftTarget, right_eye, left_eye, last_eye, numLeft, numRight)
                if eye_sect is not None:
                    last_eye = eye_sect

                # pass through functions from animation.py to get frame of animation
                attributes[2] = get_pupil_pos(last_eye)
                image = generate_frame(c, scale, attributes, images)

                cv2.imshow("Character", image) # window for character animation
                
                # TODO: integrate into function for classification and drawing 
                cv2.imshow("testR", right_eye) # window for right color right eye with drawn regions
                cv2.imshow("testL", left_eye) # window for left color left eye with drawn regions
                leye_videoWriter.write(left_eye) # for creating demo videos  
                reye_videoWriter.write(right_eye) # for creating demo videos  

                # character animation demo video
                image = cv2.resize(image, (int(image_width/2), image_height), interpolation= cv2.INTER_CUBIC)
                character_videoWriter.write(image)

            key_pressed = cv2.waitKey(10) & 0xFF
            if key_pressed == 27:
                break  # Quit on ESC
            if use_delay:
                time.sleep(.1)


    #after exiting while loop, convert to video

    print("all done")
    face_videoWriter.release()
    reye_videoWriter.release()
    leye_videoWriter.release()
    breye_videoWriter.release()
    bleye_videoWriter.release()
    character_videoWriter.release()


if __name__ == "__main__":
    main()

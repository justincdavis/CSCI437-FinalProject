import cv2
import numpy as np
import sys, os, time
from PIL import Image
from animation import Character, init_Character, generate_frame

# https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape/56909036
# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta

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
        _, binaryImg = cv2.threshold(eyeImage, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        filtered_img = cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE, kernel)
        filtered_img = cv2.morphologyEx(filtered_img, cv2.MORPH_OPEN, kernel)
        return filtered_img
    return eyeImage

# runs cv2.connectedComponentsWithStats
# then inverts binary image and runs again
# returns the actual components for normal and inverted
def getConComps(eyeImage, connected=8):
    # connected components for white blobs
    bin_comps, output, stats, centroids = cv2.connectedComponentsWithStats(eyeImage, connected)
    # connected components for former black blobs
    inverted_img = cv2.bitwise_not(eyeImage)
    i_bin_comps, i_output, i_stats, i_centroids = cv2.connectedComponentsWithStats(inverted_img, connected)
    return centroids, i_centroids

# takes the connected components for normal and inverted image, with optional parameter threshold
def findPossibleTargets(whiteBlobs, blackBlobs, threshold=2.0):
    possibleTargets = {}
    minDist = 999999
    for white_point in whiteBlobs:
        for black_point in blackBlobs:
            dist = cv2.norm(white_point - black_point, cv2.NORM_L2)
            if (dist < threshold):
                possibleTargets[dist] = white_point
                if (dist < minDist):
                    minDist = dist
    return possibleTargets, minDist

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
            print("No eyes detected!")
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
    got_image, bgr_image = camera.read()
    use_delay = False
    if not got_image:
        camera = cv2.VideoCapture("output.avi")
        use_delay = True
    # image_height, image_width, _ = bgr_image.shape
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # videoWriter = cv2.VideoWriter("output.avi", fourcc=fourcc, fps=20.0,
    #                               frameSize=(image_width, image_height))

    #create variables for character and drawing the character
    c = Character(init_Character())
    scale = 10
    # max values: 0, 13,36,7, 0, 11
    attributes = [0, 0, 0, 0, 0, 0]
    images = []

    last_face = None
    while True:
        got_image, bgr_image = camera.read()
        if not got_image:
            sys.exit()

        face_image, face = crop2Face(bgr_image, face_detector)
        if face_image is not None and face is not None:
            # videoWriter.write(bgr_image)
            eyes, eye_image = detectEyes(face_image, eye_detector, draw=True)
            right_eye, left_eye = crop2Eyes(eye_image, eyes)
            if right_eye is not None and left_eye is not None:
                image = generate_frame(c, scale, attributes, images)

                cv2.imshow("output.png", image)
                cv2.imshow("camera feed", bgr_image)
                cv2.imshow("testR", right_eye)
                cv2.imshow("testL", left_eye)

                # this is where classify pupils will go
                eyeQuads = classifyPupils(left_eye, right_eye, debug=True)

            cv2.waitKey(0)
            # key_pressed = cv2.waitKey(10) & 0xFF
            # if key_pressed == 27:
            #     break  # Quit on ESC
            # if use_delay:
            #     time.sleep(.1)


    #after exiting while loop, read images in array and convert to video

    print("all done")
    # videoWriter.release()


if __name__ == "__main__":
    main() 

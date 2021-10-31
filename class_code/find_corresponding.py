import cv2

image_file_names = ["pavilionLeft.jpg", "pavilionCenter.jpg", "pavilionRight.jpg"]

def main():
    bgr_A = cv2.imread(image_file_names[0])      # Read images
    bgr_B = cv2.imread(image_file_names[1])

    # Create two lists.  The (x,y) points go in these lists.
    ptsA = []
    ptsB = []

    # Display images.
    displayA = bgr_A.copy()
    displayB = bgr_B.copy()
    create_named_window("Image A", displayA)
    create_named_window("Image B", displayB)
    cv2.imshow("Image A", displayA)
    cv2.imshow("Image B", displayB)

    # Assign the mouse callback function, which collects (x,y) points.
    cv2.setMouseCallback("Image A", on_mouse=get_xy, param=("Image A", displayA, ptsA))
    cv2.setMouseCallback("Image B", on_mouse=get_xy, param=("Image B", displayB, ptsB))

    # Loop until user hits the ESC key.
    print("Click on points.  Hit ESC to exit.")
    while True:
        if cv2.waitKey(100) == 27:      # ESC is ASCII code 27
            if not len(ptsA) == len(ptsB):
                print("Error: you need same number of points in both images!")
            else:
                break
    print("PtsA:", ptsA)        # Print points to the console
    print("PtsB:", ptsB)


# Utility function to create an image window.
def create_named_window(window_name, image):
    # WINDOW_NORMAL allows resize; use WINDOW_AUTOSIZE for no resize.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h = image.shape[0]  # image height
    w = image.shape[1]  # image width

    # Shrink the window if it is too big (exceeds some maximum size).
    WIN_MAX_SIZE = 1000
    if max(w, h) > WIN_MAX_SIZE:
        scale = WIN_MAX_SIZE / max(w, h)
    else:
        scale = 1
    cv2.resizeWindow(winname=window_name, width=int(w * scale), height=int(h * scale))

# Mouse callback function. Appends the x,y location of mouse click to a list.
def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        window_name, image, point_list = param  # Unpack parameters
        cv2.rectangle(image, pt1=(x-15, y-15), pt2=(x+15, y+15), color=(0,0,255),
                      thickness=3)
        cv2.putText(image, str(len(point_list)), org=(x,y-15), color=(0,0,255),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, thickness=2)
        cv2.imshow(window_name, image)
        point_list.append((x,y))

if __name__ == "__main__":
    main()

# Importing necessary libraries
# Importing openCV for computer vision tasks
import cv2
# Hand tacking module for detecting hands in the frame
from cvzone.HandTrackingModule import HandDetector
# For numerical operations
import numpy as np
# Mathematical functions
import math
# Importing time module for adding delays or measuring time intervals
import time


def capture_and_save_image(camera, detector, folder_path, image_size=300, offset=20):
    # Read a frame from the camera
    success, frame = camera.read()
    
    # Detect hands in the frame
    detected_hands, frame = Hand_detector.findHands(frame)
    
    # Check if hands are detected
    if detected_hands:
        # Get the first detected hand
        hand = detected_hands[0]
        # Get the bounding box coordinates of the ditected hand
        x, y, w, h = hand['bbox']
        
        # Create a white canvas for resizing the cropped image
        White_Img = np.ones((image_size, image_size, 3), np.uint8) * 255
        
        # Crop the region around the hand
        Cropped_Img = frame[y - offset:y + h + offset, x - offset:x + w + offset]

        # Calculate aspect ratio of the hand bounding box
        aspect_ratio = h / w

        # Resize the cropped image while maintaining aspect ratio
        if aspect_ratio > 1:
            # resizing factor
            k = image_size / h
            # width after resize
            width_calculated = math.ceil(k * w)
            # resize the image
            image_resize = cv2.resize(Cropped_Img, (width_calculated, image_size))
            # Calculate gap for centering
            width_gap = math.ceil((image_size - width_calculated) / 2)
            # Resize image on the white canvas
            White_Img[:, width_gap:width_calculated + width_gap] = image_resize
        else:
            # resizing factor
            k = image_size / w
            # height after resize
            height_calculated = math.ceil(k * h)
            # resize the image
            image_resize = cv2.resize(Cropped_Img, (image_size, height_calculated))
            # Calculate gap for centering
            height_gap = math.ceil((image_size - height_calculated) / 2)
            # Resize image on the white canvas
            White_Img[height_gap:height_calculated + height_gap, :] = image_resize

        # Display the cropped and resized images
        cv2.imshow("ImageCrop", Cropped_Img)
        cv2.imshow("ImageWhite", White_Img)

    # Display the original image
    cv2.imshow("Image", frame)

    # Check for key press 's' to save the image
    key = cv2.waitKey(1)
    if key == ord("s"):
        save_image(folder_path, White_Img)
    
    


# Custom Dataset
# Saving the image in the folder
def save_image(folder_path, White_Img):
    global counter
    counter += 1
    cv2.imwrite(f'{folder_path}/Image_{time.time()}.jpg', White_Img)
    print(counter)


# Initialize variables
camera = cv2.VideoCapture(0)
# hand Detection using Hand Detector 
Hand_detector = HandDetector(maxHands=1)
# folder path for respected attribute
folder_path = "Data/A"
# maintain so recognizable ammount of count
counter = 0

# Main loop
while True:
    capture_and_save_image(camera, Hand_detector, folder_path)

# Importing necessary libraries
# Importing openCV for computer vision tasks
import cv2
# Hand tacking module for detecting hands in the frame
from cvzone.HandTrackingModule import HandDetector
# Classifier module for hand gesture classification
from cvzone.ClassificationModule import Classifier
# For numerical operations
import numpy as np
# Mathematical functions
import math

# Initialize video capture, hand detector, and classifier
# Connection to the default camera (index 0)
camera = cv2.VideoCapture(0)
# Hand detector to track a single hand
Hand_detector = HandDetector(maxHands=1)
# Pre-trained hand gesture classification modeland labels 
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Constants for image processing
# Cropping hand region
offset = 20
# Resiz image for classification
imgSize = 300

# List of labels for classification
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "yo", "super", "love", "ok", "NotOK"]

while True:
    # Read a frame from the camera
    # Capture a frame from the video feed
    success, frame = camera.read()
    # Creating a copy of the frame for output visualization
    imgOutput = frame.copy()
    
    # Detect hands in the frame
    detected_hands, frame = Hand_detector.findHands(frame)
    
    if detected_hands:
        # Select the first hand that is detected
        hand = detected_hands[0]
        # Get the bounding box coordinates of the ditected hand
        x, y, w, h = hand['bbox']

        # Create a white canvas for resizing the cropped image
        # Create a white image of specific size
        White_img = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        # crop hand region from the frame
        CroppedImg = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        
        # shape of the cropped image
        imgCropShape = CroppedImg.shape

        # aspect ratio of the cropped image 
        aspectRatio = h / w

        # Resize the cropped image while maintaining aspect ratio
        if aspectRatio > 1:
            # resizing factor
            k = imgSize / h
            # width after resize
            widthCalculated = math.ceil(k * w)
            # resize the image
            imgResize = cv2.resize(CroppedImg, (widthCalculated, imgSize))
            # shape of the resized image
            imgResizeShape = imgResize.shape
            # Calculate gap for centering
            widthGap = math.ceil((imgSize - widthCalculated) / 2)
            # Resize image on the white canvas
            White_img[:, widthGap:widthCalculated + widthGap] = imgResize
            # get Hand gesture prediction from the classifier
            prediction, index = classifier.getPrediction(White_img, draw=False)
            # print the predicted label and index 
            print(prediction, index)

        else:
            # resizing factor
            k = imgSize / w
            # height after resize
            heightCalculated = math.ceil(k * h)
            # resize the image
            imgResize = cv2.resize(CroppedImg, (imgSize, heightCalculated))
            # shape of the resized image
            imgResizeShape = imgResize.shape
            # Calculate gap for centering
            heightGap = math.ceil((imgSize - heightCalculated) / 2)
            # Resize image on the white canvas
            White_img[heightGap:heightCalculated + heightGap, :] = imgResize
            # get Hand gesture prediction from the classifier
            prediction, index = classifier.getPrediction(White_img, draw=False)

        # Display rectangles and text on the output image
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        # Display predicted label
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        # rectangle around hand
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)

        # Display the full-size camera window
        cv2.imshow("Image", cv2.resize(imgOutput, (800, 600)))

        cv2.imshow("CroppedImage", CroppedImg)
        cv2.imshow("WhiteImage", White_img)

    else:
        # Display the full-size camera window when no hands are detected
        cv2.imshow("Image", cv2.resize(imgOutput, (800, 600)))

    # Press 'Esc' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break
    # press q to quit the game
    elif cv2.waitKey(1) == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()

#importing necessary libraries
# To pick AI decision randomly
import random
# For computer vision tasks 
import cv2
# additional computer vision utilities like overlayPNG,etc.
import cvzone
# importing handDetector module to detect hands using mediapipe
from cvzone.HandTrackingModule import HandDetector
# for proper synchronization and timer
import time

#setting up the video capture

#Initializing video capture object, using the default camera (index 0)
camera = cv2.VideoCapture(0)
#width 640 pixels
camera.set(3, 640)
#height 480 pixels
camera.set(4, 480)

#Initalizing HandDetector with a maximum of 1 hand to be detected as the game is played through single hand
hand_detector = HandDetector(maxHands=1)

#initialization of necessary variables
#Initally timer should be at 0
timer = 0
# Initalizing lists for scores [AI, human_sign]
scores = [0, 0]
result_state = False
game_started = False

#Main game
while True:
    #Background Image
    background_Img = cv2.imread("Images/Background.png")
    #reading a frame form the video capture
    success, frame = camera.read()
    
    #resizing the video capture frame according to the background image that we made
    frame_Scaled = cv2.resize(frame, (0, 0), None, 0.875, 0.875)
    #croping the video capture so that it can shrink even to both sides 
    frame_Scaled = frame_Scaled[:, 80:480]

    #Finding Hands using the HandDetector
    # it will also update which hand it is detected (right/left)
    hands, frame = hand_detector.findHands(frame_Scaled)

    #let's game_started the game
    if game_started:
        #checking result state
        if not result_state :
            #timer setup
            timer = time.time() - initialTime
            #timer color , font etc properties
            cv2.putText(background_Img, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 4)
            
            #let's set the timer for 3 seconds and then update the result_state and timer
            if timer > 3:
                result_state = True
                timer = 0
                
                #check the hands
                if hands:
                    human_sign = None
                    detected_hand = hands[0]
                    fingers_count = hand_detector.fingersUp(detected_hand)
                    # Rock
                    if fingers_count == [0, 0, 0, 0, 0]:
                        human_sign = 1
                    # Paper
                    if fingers_count == [1, 1, 1, 1, 1]:
                        human_sign = 2
                    # Scissors
                    if fingers_count == [0, 1, 1, 0, 0]:
                        human_sign = 3
                    
                    # Generating a random mover for the AI
                    AI_move = random.choice(['rock','paper','scissors'])
                    imgAI = cv2.imread(f'Images/{AI_move}.png', cv2.IMREAD_UNCHANGED)
                    background_Img = cvzone.overlayPNG(background_Img, imgAI, (149, 310))

                    #if human_sign sign is rock and AI is scissors, if human_sign sign is paper and AI is rock and human_sign sign is scissors and AI is paper
                    if (human_sign == 1 and AI_move == 'scissors') or (human_sign == 2 and AI_move == 'rock') or (human_sign == 3 and AI_move == 'paper'):
                        scores[1] += 1 #human_sign wins
                        
                    #same game logic when AI wins the game
                    if (human_sign == 3 and AI_move == 'rock') or (human_sign == 1 and AI_move == 'paper') or (human_sign == 2 and AI_move == 'scissors'):
                        scores[0] += 1

    #overlaying the video capture according to the background image
    background_Img[240:660, 796:1196] = frame_Scaled
    
    # overlaying the AI signs according to the frame
    if result_state:
        background_Img = cvzone.overlayPNG(background_Img, imgAI, (65, 185))

    # Display the scores
    cv2.putText(background_Img, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
    cv2.putText(background_Img, str(scores[1]), (1122, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

    # Display background image with overlay
    cv2.imshow("BG", background_Img)
    
    # press s to game_started the game
    key = cv2.waitKey(1)
    if key == ord('s'):
        game_started = True
        initialTime = time.time()
        result_state = False
    # press q to quit the game
    elif key == ord('q'):
        break
        
#close 
camera.release()
cv2.destroyAllWindows()
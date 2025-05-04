import cv2 
import mediapipe as mp 
import math
from google.protobuf.json_format import MessageToDict 

mpHands = mp.solutions.hands 
hands = mpHands.Hands( 
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.4, 
    max_num_hands=2) 
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) 

while True:     
    success, img = cap.read() 
    img = cv2.flip(img, 1) 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    results = hands.process(imgRGB) 
    if results.multi_hand_landmarks: 
        fingertip_indexes = [4, 8, 12, 16, 20,0]
        for hand_landmarks in results.multi_hand_landmarks:
            """mp_drawing.draw_landmarks(
                img, 
                hand_landmarks, 
                mpHands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))""" #skeleton
            for i in fingertip_indexes:
                landmark_x = int(hand_landmarks.landmark[i].x * img.shape[1])
                landmark_y = int(hand_landmarks.landmark[i].y * img.shape[0])
                wrist_x = int(hand_landmarks.landmark[0].x * img.shape[1])
                wrist_y = int(hand_landmarks.landmark[0].y * img.shape[0])
                thumb_x = int(hand_landmarks.landmark[4].x * img.shape[1])
                thumb_y = int(hand_landmarks.landmark[4].y * img.shape[0])
                cv2.circle(img, (landmark_x, landmark_y), 10, (255, 0, 0), 1)
                cv2.line(img, (landmark_x, landmark_y), (wrist_x, wrist_y), (0, 255, 0), 1)
                cv2.line(img, (landmark_x, landmark_y), (thumb_x, thumb_y), (0, 255, 0), 1)
                dist=math.dist((landmark_x, landmark_y), (thumb_x, thumb_y))
                thumb_midpoint=((thumb_x + landmark_x) // 2, (thumb_y + landmark_y) // 2)
                cv2.putText(img, str(int(dist)), thumb_midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                wrist_midpoint=((wrist_x + landmark_x) // 2, (wrist_y + landmark_y) // 2)
                cv2.putText(img, str(int(dist)), wrist_midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                for i in results.multi_handedness: 
                    dict_result = MessageToDict(i)
                    if 'classification' in dict_result and len(dict_result['classification']) > 0:
                        label = dict_result['classification'][0].get('label', '')
                        if label == 'Left': 
                            cv2.putText(img, "left", (wrist_x+20,wrist_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        elif label == 'Right': 
                            cv2.putText(img, "right", (wrist_x+20,wrist_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        
                        
            
    height, width,_ = img.shape
    height*=1.5
    width*=1.5
    img=cv2.resize(img,(int(width),int(height)))
    cv2.imshow('Image', img) 
    if cv2.waitKey(1) & 0xff == ord('q'): 
        break
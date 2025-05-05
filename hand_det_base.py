import cv2 
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

class LandmarkValues():
    def __init__(self, distance=None,l1=None,l2=None,hand=None):
        self.distance = distance
        self.l1 = l1
        self.l2 = l2
        self.hand = hand
      

    def __repr__(self):
        return f"LandmarkValues(Dist={self.distance}, l1={self.l1}, l2={self.l2}, hand={self.hand})"

mpHands = mp.solutions.hands 
hands = mpHands.Hands( 
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.4, 
    max_num_hands=2) 
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) 

def get_invariant_ratio(landmarks, a_idx, b_idx, ref1_idx=0, ref2_idx=9):
    a = landmarks[a_idx]
    b = landmarks[b_idx]
    ref1 = landmarks[ref1_idx]
    ref2 = landmarks[ref2_idx]
    
    # Compute the two distances
    dist_ab = ((b.x - a.x)**2 + (b.y - a.y)**2) ** 0.5
    dist_ref = ((ref2.x - ref1.x)**2 + (ref2.y - ref1.y)**2) ** 0.5
    
    return dist_ab / (dist_ref + 1e-6)  # Avoid division by zero

fingertip_indexes = [4, 8, 12, 16, 20] #fingertips and wrist
wrist_index=0
thumb_index=4

def get_values_for_hand(landmarks,hand):
    temp_list=[]
    for i in fingertip_indexes:
        temp_thumb=LandmarkValues(hand=hand)
        temp_wrist=LandmarkValues(hand=hand)
        landmark_x = int(landmarks[i].x * img.shape[1])
        landmark_y = int(landmarks[i].y * img.shape[0])
        wrist_x = int(landmarks[0].x * img.shape[1])
        wrist_y = int(landmarks[0].y * img.shape[0])
        thumb_x = int(landmarks[4].x * img.shape[1])
        thumb_y = int(landmarks[4].y * img.shape[0])
        cv2.circle(img, (landmark_x, landmark_y), 10, (255, 0, 0), 1)
        cv2.line(img, (landmark_x, landmark_y), (wrist_x, wrist_y), (0, 255, 0), 1)
        cv2.line(img, (landmark_x, landmark_y), (thumb_x, thumb_y), (0, 255, 0), 1) #paths betweeen key landmarks
        thumb_dist=get_invariant_ratio(landmarks, i, thumb_index) * 100
        wrist_dist=get_invariant_ratio(landmarks, i, wrist_index) * 100
        thumb_midpoint=((thumb_x + landmark_x) // 2, (thumb_y + landmark_y) // 2)
        cv2.putText(img, str(int(thumb_dist)), thumb_midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        wrist_midpoint=((wrist_x + landmark_x) // 2, (wrist_y + landmark_y) // 2)
        cv2.putText(img, str(int(wrist_dist)), wrist_midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        temp_thumb.distance=thumb_dist
        temp_wrist.distance=wrist_dist
        temp_thumb.l1="thumb"
        temp_wrist.l1="wrist"
        temp_thumb.l2=i
        temp_wrist.l2=i
        temp_list.append(temp_thumb)
        temp_list.append(temp_wrist)
        
    return temp_list


while True:     
    values=[]
    success, img = cap.read() 
    img = cv2.flip(img, 1) 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks: 
        for i in results.multi_handedness:
            dict_result = MessageToDict(i)
            if 'classification' in dict_result and len(dict_result['classification']) > 0:
                label = dict_result['classification'][0].get('label', '')
                if len(results.multi_hand_landmarks) != 2:
                    if label == 'Left': 
                        values+=get_values_for_hand(results.multi_hand_landmarks[0].landmark,label)    
                    if label == 'Right':
                        values+=get_values_for_hand(results.multi_hand_landmarks[0].landmark,label)
                elif len(results.multi_hand_landmarks) == 2:
                    if label == 'Left': 
                        values+=get_values_for_hand(results.multi_hand_landmarks[0].landmark,label)    
                    if label == 'Right':
                        pass 
                        values+=get_values_for_hand(results.multi_hand_landmarks[1].landmark,label)
    """for i in values:
        print(i)"""
    height, width,_ = img.shape
    height*=1.5
    width*=1.5
    img=cv2.resize(img,(int(width),int(height)))
    cv2.imshow('Image', img) 
    if cv2.waitKey(1) & 0xff == ord('q'): 
        break
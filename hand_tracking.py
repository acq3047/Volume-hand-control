# Download the packages

import cv2
import mediapipe as mp
import math

print(dir(mp))

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.max_hands,
                                        min_detection_confidence=self.detection_conf,
                                        min_tracking_confidence=self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils
        self.lm_list = []
        self.tipIds = [4, 8, 12, 16, 20]


    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks,
                                               self.mphands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        self.lm_list = []
        bbox = (0, 0, 0, 0)
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            x_list = []
            y_list = []
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            bbox = (xmin, ymin, xmax, ymax)
            if draw:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        return self.lm_list, bbox
    
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lm_list[self.tipIds[0]][1] > self.lm_list[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lm_list[self.tipIds[id]][2] < self.lm_list[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def find_distance(self, pt1, pt2, img, draw=True):
        x1, y1 = self.lm_list[pt1][1], self.lm_list[pt1][2]
        x2, y2 = self.lm_list[pt2][1], self.lm_list[pt2][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 165, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 165, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 240), 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

        len_line = math.hypot(x2 - x1, y2 - y1)
        return len_line, img, [x1, y1, x2, y2, cx, cy]


import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConf, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, handNo=0, draw=True):  # only for a particular hand
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
        return lm_list

    def hand_bbox(self, pts):
        """compute the bounding box of the hand"""
        array_pt = np.array(pts)
        min_x = min(array_pt[:, 1])
        min_y = min(array_pt[:, 2])
        max_x = max(array_pt[:, 1])
        max_y = max(array_pt[:, 2])
        return [(min_x, min_y), (max_x, max_y)]








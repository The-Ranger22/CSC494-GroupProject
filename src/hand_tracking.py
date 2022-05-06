'''
Hand tracking using mediapipe and cv2 libraries

reference: https://stackoverflow.com/questions/66876906/create-a-rectangle-around-all-the-points-returned-from-mediapipe-hand-landmark-d
'''

# Courtesy of Dr. Zhou

import cv2
import mediapipe as mp

from tensorflow.image import resize
mphands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

class HandFinder:
    def __init__(self):
        self._hands = mphands.Hands()
        self._pixel_offset = 20

    def detect(self, image, draw=True, bounding_box=True, skeleton=False):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._hands.process(imageRGB)
        h, w, c = image.shape
        rects = []
        if results.multi_hand_landmarks:
            for handLMs in results.multi_hand_landmarks:
                # find the ‘x’ and ‘y’ coordinates of each hand point
                # and get the bounding box coordinates of each hand as well
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y

                    rects.append([
                        id,
                        x_min - self._pixel_offset,
                        x_max + self._pixel_offset,
                        y_min - self._pixel_offset,
                        y_max + self._pixel_offset
                    ])

                pad_x = round((x_max - x_min) / 2)
                pad_y = round((y_max - y_min) / 2)
                padding = pad_x if pad_x > pad_y else pad_y
                center_x = x_min + round((x_max - x_min) / 2)
                center_y = y_min + round((y_max - y_min) / 2)

                x_max = center_x + padding + self._pixel_offset
                x_min = center_x - padding - self._pixel_offset

                y_max = center_y + padding + self._pixel_offset
                y_min = center_y - padding - self._pixel_offset



                if draw:
                    # Preserve original image via copy
                    hand = image.copy()
                    if bounding_box:
                        # Draw the bounding box
                        cv2.rectangle(
                            image,
                            (x_min, y_min),
                            (x_max, y_max),
                            (0, 255, 0),
                            2
                        )

                    if skeleton:
                        # Draw the hand landmarks and hand connections
                        mpDraw.draw_landmarks(image, handLMs, mphands.HAND_CONNECTIONS)
                        return (
                            hand,
                            image
                        )
                return image[y_min:y_max, x_min:x_max], image

        return rects

# def handsFinder(image, draw=True):
#     ''' This function aims to detect hands
#      It accepts image/frame from webcam
#      It returns the bounding box around the hand(s) area
#      '''
#     # Initialization
#     imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     results = hands.process(imageRGB)
#     h, w, c = image.shape
#     rects = []
#     if results.multi_hand_landmarks:
#         for handLMs in results.multi_hand_landmarks:
#             # find the ‘x’ and ‘y’ coordinates of each hand point
#             # and get the bounding box coordinates of each hand as well
#             x_max = 0
#             y_max = 0
#             x_min = w
#             y_min = h
#             for lm in handLMs.landmark:
#                 x, y = int(lm.x * w), int(lm.y * h)
#                 if x > x_max:
#                     x_max = x
#                 if x < x_min:
#                     x_min = x
#                 if y > y_max:
#                     y_max = y
#                 if y < y_min:
#                     y_min = y
#                 rects.append([id, x_min, x_max, y_min, y_max])
#
#             if draw:
#                 # Draw the bounding box
#                 cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#                 # Draw the hand landmarks and hand connections
#                 mpDraw.draw_landmarks(image, handLMs, mphands.HAND_CONNECTIONS)
#
#     return rects


if __name__ == "__main__":
    ''' Capturing an image input and processing it
    and displaying the output'''
    hand_finder = HandFinder()
    cap = cv2.VideoCapture(0)
    pixel_offset = 15
    while True:
        success, image = cap.read()
        hand_img = hand_finder.detect(image, False)
        if len(hand_img) != 0:
            try:
                print(hand_img.shape)
                hand_img = resize(hand_img, (128, 128), method='nearest').numpy()
                cv2.imshow("Output", hand_img)
            except:
                cv2.imshow("Output", image)
        else:
            cv2.imshow("Output", image)
        cv2.waitKey(1)

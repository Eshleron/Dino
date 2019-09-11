
import time

import numpy as np
import tensorflow as tf
from PIL import ImageGrab
import cv2

import pyautogui


MODEL_FILENAME = 'DINO-2-conv-64-nodes-3-dense-1568187117.model'

model = tf.keras.models.load_model(MODEL_FILENAME)


def main():
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)

    prev_score = None

    paused = False
    while True:
        if not paused:
            screen = ImageGrab.grab(bbox=(350, 500, 550, 680))
            score = ImageGrab.grab(bbox=(1700, 200, 1900, 400))
            if prev_score == score:
                score.save('scores/score_' + str(time.time()) + '.bmp', 0)
                pyautogui.press('enter')

            # screen.save('Templates_1/template_' + str(time.time()) + '.bmp', 0)
            screen = np.asarray(screen)
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80, 80))
            screen = np.expand_dims(screen, axis=0)
            screen = np.expand_dims(screen, axis=3)
            prediction = model.predict(screen)

            if prediction[0][0] >= 0.9:
                pyautogui.press('space')

            # time.sleep(.05)
            prev_score = score


main()

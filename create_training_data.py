"""
TO OPEN THE GAME: chrome://dino/

"""


import os
import time

import numpy as np
from PIL import ImageGrab
import cv2

from getkeys import key_check


FILE_NAME = 'training_data.npy'


def keys_to_output(keys):

    if ' ' in keys:
        output = [1]  # 0 value is JUMP
    else:
        output = [0]  # else GROUND
    return output


if os.path.isfile(FILE_NAME):
    print('File exists, loading previous data!')
    training_data = list(np.load(FILE_NAME))
else:
    print('File does not exist, starting fresh!')
    training_data = []


def main():
    for i in range(4, 0, -1):
        print(i)
        time.sleep(1)

    paused = False
    while True:
        if not paused:
            screen = ImageGrab.grab(bbox=(350, 500, 550, 680))
            # screen.save('Templates_1/template_' + str(time.time()) + '.bmp', 0)
            # screen.save('C:/Users/%username%/PycharmProjects/Dino/Templates_1/template_' + str(time.time()) + '.bmp', 0)
            screen = np.asarray(screen)
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80, 80))
            # cv2.imshow('test', screen)
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen, output])

            if len(training_data) % 100 == 0:
                print(len(training_data))
                np.save(FILE_NAME, training_data)

        time.sleep(.05)

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main()

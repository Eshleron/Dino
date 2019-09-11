from collections import Counter
from random import shuffle
import time

import cv2
import numpy as np
import pandas as pd


train_data = np.load('training_data.npy')


def info(data):
    df = pd.DataFrame(data)
    # print(df.head())
    print(Counter(df[1].apply(str)))


info(train_data)

ups = []
grounds = []

for data in train_data:
    img = data[0]
    choice = data[1]

    if np.average(img) < 180:
        for i, pix in enumerate(img):
            img[i] = abs(255-pix)

    if choice == [1]:
        ups.append([img, choice])
    elif choice == [0]:
        grounds.append([img, choice])
    else:
        print('no matches')


grounds = grounds[:len(ups)]
ups = ups[:len(grounds)]

final_data = ups + grounds
shuffle(final_data)

np.save('training_data_balanced.npy', final_data)

info(final_data)

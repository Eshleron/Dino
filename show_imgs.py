import time

import cv2
import numpy as np


train_data = np.load('training_data_balanced.npy')
print('length of data:', len(train_data))

for i, data in enumerate(train_data):
    img = data[0]
    choice = data[1]
    cv2.imshow('Test', img)
    time.sleep(0.2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

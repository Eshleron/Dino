# To open TENSORBOARD use: tensorboard --logdir=logs/
# or tensorboard --logdir=logs/ --host localhost --port 8088
# tensorboard --logdir=D:/PycharmProjects/Dino/logs/ --host localhost --port 8088

import time

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


DATA_FILENAME = 'training_data_balanced.npy'

dataset = np.load(DATA_FILENAME)
# print(type(dataset))

X = np.expand_dims((np.array([np.array(data[0]/255) for data in dataset])), axis=3)
# print(type(X))
print(X.shape)
# print(X[1])

y = np.array([np.array(data[1]) for data in dataset])
# print(y)


dense_layers = [7]
layer_sizes = [2, 4, 8, 16, 32, 64, 128]
conv_layers = [1, 2, 3, 4]  # 3 16 4 was BEST

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=(80, 80, 1)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            model.fit(X, y,
                      batch_size=100,
                      epochs=10,
                      validation_split=0.2,
                      callbacks=[tensorboard])

model.save('DINO-{}.model'.format(NAME))

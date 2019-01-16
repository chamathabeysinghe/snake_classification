import numpy.random as rng
from keras import backend as K
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam


class SiameseModel:
    input_shape = (64, 64, 3)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    def __init__(self):
        pass

    def build(self):
        input_shape = (64, 64, 3)
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        convnet = Sequential()
        convnet.add(Conv2D(10, (8, 8), activation='relu', input_shape=input_shape))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(10, (4, 4), activation='relu'))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(10, (4, 4), activation='relu'))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(10, (4, 4), activation='relu'))
        convnet.add(Flatten())
        convnet.add(Dense(10, activation='sigmoid'))

        encoded_l = convnet(left_input)
        encoded_r = convnet(right_input)

        L1_distance = lambda x: K.abs((x[0] - x[1]))
        both = merge([encoded_l, encoded_r], mode=L1_distance, output_shape=lambda x: x[0])
        prediction = Dense(1, activation='sigmoid')(both)
        siamese_net = Model(input=[left_input, right_input], outputs=prediction)
        return siamese_net

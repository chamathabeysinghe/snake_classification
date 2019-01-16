import numpy.random as rng
from keras import backend as K
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
from networks.siamese_network.SiameseLoader import SiameseLoader

w = 150
h = 150
d = 3

def W_init(shape, name=None):
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)


input_shape = (w, h, d)
left_input = Input(input_shape)
right_input = Input(input_shape)

convnet = Sequential()
convnet.add(Conv2D(10,(8,8), activation='relu', input_shape=input_shape))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(10,(4,4), activation='relu'))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(10,(4,4),activation='relu'))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(10, (4,4), activation='relu'))
convnet.add(Flatten())
convnet.add(Dense(10, activation='sigmoid'))

encoded_l = convnet(left_input)
encoded_r = convnet(right_input)

L1_distance = lambda x: K.abs((x[0]-x[1]))
both = merge([encoded_l, encoded_r], mode=L1_distance, output_shape=lambda x: x[0])
prediction = Dense(1, activation='sigmoid')(both)
siamese_net = Model(input=[left_input, right_input], outputs=prediction)

optimizer = Adam(0.00006)
siamese_net.compile(loss='binary_crossentropy', optimizer=optimizer)
siamese_net.summary(line_length=150)

# image_loader = SiameseLoader('./data')


sample_1 = np.random.random((100,w,h,d))
sample_2 = np.random.random((100,w,h,d))
targets = np.zeros((100,))
targets[50:] = 1

siamese_net.fit([sample_1, sample_2], targets, batch_size=32, epochs=10)

# siamese_net.fit_generator(image_loader.generate(32), steps_per_epoch=4, epochs=2000,
#                           validation_data=image_loader.generate_val(), validation_steps=1)


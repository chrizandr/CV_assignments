from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.preprocessing import image
from keras import backend as K
from keras.layers.core import Lambda
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.models import Model

import numpy as np
import pdb
import os

def batch_generator(x_path, y_path, batch_size=32):
    files = os.listdir(x_path)

    file = open(y_path, 'r').read()
    y = file.strip().split(',')

    X_batch = np.zeros((batch_size, 227, 227, 3))
    y_batch = np.zeros((batch_size, 8))

    while True:
        for i in range(batch_size):
            index = np.random.randint(0, high=len(files))
            file = str(index+1)+'.jpg'
            img = image.load_img(x_path+file, target_size=(227, 227))
            img = image.img_to_array(img, data_format='channels_first')
            X_batch[i] = img
            y_batch[i][int(y[index])-1] = 1
        yield X_batch, y_batch


def AlexNet(weights_path=None):
    inputs = Input(shape=(3, 227, 227))

    conv_1 = Convolution2D(96, (11, 11), strides=(4, 4), activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = concatenate([
        Convolution2D(128, (5, 5), activation="relu", name='conv_2_'+str(i+1))(
            splittensor(ratio_split=2, id_split=i)(conv_2)
        ) for i in range(2)], axis=1, name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, (3, 3), activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = concatenate([
        Convolution2D(192, (3, 3), activation="relu", name='conv_4_'+str(i+1))(
            splittensor(ratio_split=2, id_split=i)(conv_4)
        ) for i in range(2)], axis=1, name="conv_4")

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = concatenate([Convolution2D(128, (3, 3), activation="relu", name='conv_5_'+str(i+1))
                          (splittensor(ratio_split=2, id_split=i)(conv_5)) for i in range(2)
                          ],
                         axis=1, name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_3')(dense_3)
    prediction = Activation("softmax", name="softmax")(dense_3)

    model = Model(inputs=inputs, outputs=prediction)
    if weights_path:
        model.load_weights(weights_path)

    if K.backend() == 'tensorflow':
        convert_all_kernels_in_model(model)
    return model


def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """Cross channel normalization in the original Alexnet."""
    def f(X):
        K.set_image_dim_ordering('th')
        if K.backend() == 'tensorflow':
            b, ch, r, c = X.get_shape()
        else:
            b, ch, r, c = X.shape
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1)),
                                                                  ((0, 0), (half, half)))
        extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
        scale = k
        for i in range(n):
            if K.backend() == 'tensorflow':
                ch = int(ch)
            scale += alpha * extra_channels[:, i:i+ch, :, :]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        if K.backend() == 'tensorflow':
            div = int(X.get_shape()[axis]) // ratio_split
        else:
            div = X.shape[axis] // ratio_split

        if axis == 0:
            output = X[id_split*div:(id_split+1)*div, :, :, :]
        elif axis == 1:
            output = X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:, :, id_split*div:(id_split+1)*div, :]
        elif axis == 3:
            output = X[:, :, :, id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)


if __name__ == '__main__':
    alexmodel = AlexNet('data/alexnet_weights.h5')
    for layer in alexmodel.layers:
        layer.trainable = False

    final_dense = Dropout(0.5)(alexmodel.get_layer('dense_2').output)
    final_dense = Dense(8, name='final_dense')(final_dense)
    prediction = Activation('softmax', name='softmax')(final_dense)

    model = Model(inputs=alexmodel.get_layer('conv_1').input, outputs=prediction)
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    pdb.set_trace()
    model.fit_generator(batch_generator('data/hw2_data/train/', 'data/hw2_data/train.csv', 20), steps_per_epoch=10, epochs=20)

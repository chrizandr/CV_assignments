from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
from keras.preprocessing import image
from keras.models import Model

import numpy as np
import pdb
import os

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs



def VGG_Writer(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling='', classes=8):
    """VVG16 without final dense layers."""
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=False)
    # pdb.set_trace()
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=True)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=True)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=True)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=True)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=True)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=True)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=True)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=True)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)



    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    feature_part = Model(inputs, x, name='vgg16')

    weights_path = 'vgg16_weights.h5'

    feature_part.load_weights(weights_path)

    for layer in feature_part.layers:
        layer.trainable = False

    #img_input = Input(shape=(224, 224, 3))
    #features = feature_part(img_input)
    #x = Flatten(name='flatten', trainable=True)(features)
    #x = Dense(4096, activation='relu', name='fc1', trainable=True)(x)
    #x = Dense(4096, activation='relu', name='fc2', trainable=True)(x)
    #predictions = Dense(classes, activation='softmax', name='predictions', trainable=True)(x)

    #model = Model(img_input, predictions, name="classifier")
    return feature_part


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


if __name__ == '__main__':
    input_image = Input(shape=(224, 224, 3))
    vgg_model = VGG_Writer(input_tensor=input_image) #Y
    for layer in vgg_model.layers:
        layer.trainable = False
    pdb.set_trace()
    '''#final_dense = Dropout(0.5)(vgg_model.get_layer('dense_2').output)
    final_dense = Dense(8, name='final_dense')(final_dense)
    prediction = Activation('softmax', name='softmax')(final_dense)

    model = Model(inputs=alexmodel.get_layer('conv_1').input, outputs=prediction)
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    pdb.set_trace()
    model.fit_generator(batch_generator('data/hw2_data/train/', 'data/hw2_data/train.csv', 20), steps_per_epoch=10, epochs=20)
    '''

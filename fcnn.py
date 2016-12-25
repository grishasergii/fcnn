from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.regularizers import l2


def fcnet(nb_classses, nb_channels):
    """
    Constructs a fully convolutional neural network model
    :param nb_classses: int, number of classes
    :param nb_channels: int, number of channels in input images (1 - grayscale, 3 - rgb)
    :return: instance of Keras Sequential model
    """

    l2_norm = l2(0.01)

    model = Sequential()

    model.add(Convolution2D(64, 3, 3, input_shape=(nb_channels, None, None), border_mode='same', W_regularizer=l2_norm))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2_norm))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, border_mode='same', W_regularizer=l2_norm))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', W_regularizer=l2_norm))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, 1, 1, W_regularizer=l2_norm))
    model.add(Activation('relu'))

    model.add(Convolution2D(8, 1, 1, W_regularizer=l2_norm))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_classses, 1, 1, W_regularizer=l2_norm))
    model.add(Activation('relu'))

    model.add(GlobalAveragePooling2D())

    model.add(Activation('softmax'))

    return model

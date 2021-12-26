# -*- coding: utf-8 -*-
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Activation, Input
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras.models import Model


def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True, watermark=None):
    if watermark:
        if watermark.no_embed == True:
            conv = Conv2D(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          name='watermark_layer')
        else:
            conv = Conv2D(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=watermark,
                          name='watermark_layer')
    else:
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, watermark=None, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in)')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            if watermark and watermark.target_block - 1 == stack and res_block == 0:
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None,
                                 watermark=watermark)
            else:
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


class CIFAR10:
    @staticmethod
    def load_data():
        (training_images, training_labels), (test_images, test_labels) = cifar10.load_data()
        training_labels = to_categorical(training_labels, 10)
        test_labels = to_categorical(test_labels, 10)
        return training_images, training_labels, test_images, test_labels

    @staticmethod
    def get_model(shape, regularizer):
        return resnet_v1(shape, depth=20, watermark=regularizer)

    @staticmethod
    def get_validate_model(shape, model, regularizer):
        regularizer.no_embed = True
        validate_model = CIFAR10.get_model(shape, regularizer)
        validate_model.set_weights(model.get_weights())
        return validate_model

    
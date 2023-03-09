# -*- coding: utf-8 -*-
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense


class MNIST:
    @staticmethod
    def load_data():
        (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
        training_images = training_images.reshape(60000, 28, 28, 1)
        test_images = test_images.reshape(10000, 28, 28, 1)
        training_labels = to_categorical(training_labels, 10)
        test_labels = to_categorical(test_labels, 10)
        return training_images, training_labels, test_images, test_labels

    @staticmethod
    def get_model(shape, regularizer):
        model = Sequential()

        model.add(Conv2D(6, (5, 5), activation='relu', padding='same', name='block1_conv1', input_shape=shape))
        model.add(MaxPool2D(pool_size=(2, 2), name='block1_pool1'))
        if regularizer.no_embed == True:
            model.add(Conv2D(16, (5, 5), activation='relu', padding='same', name='watermark_layer'))
        else:
            model.add(Conv2D(16, (5, 5), activation='relu', padding='same', name='watermark_layer',
                             kernel_regularizer=regularizer))
        model.add(MaxPool2D(pool_size=(2, 2), name='block2_pool1'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(120, activation='relu', name='fc1'))
        model.add(Dense(84, activation='relu', name='fc2'))
        model.add(Dense(10, name='before_softmax', activation='softmax'))

        return model

    @staticmethod
    def get_validate_model(shape, model, regularizer):
        regularizer.no_embed = True
        validate_model = MNIST.get_model(shape, regularizer)
        validate_model.set_weights(model.get_weights())
        return validate_model

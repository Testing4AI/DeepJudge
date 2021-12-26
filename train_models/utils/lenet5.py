from tensorflow import keras


def Lenet5(shape, num_classes=10):
    input_tensor = keras.layers.Input(shape=shape)
    x = keras.layers.Convolution2D(6, (5, 5), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    x = keras.layers.Convolution2D(16, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(120, activation='relu', name='fc1')(x)
    x = keras.layers.Dense(84, activation='relu', name='fc2')(x)
    x = keras.layers.Dense(num_classes, name='before_softmax')(x)
    x = keras.layers.Activation('softmax', name='predictions')(x)

    return keras.models.Model(input_tensor, x)


def Lenet1(shape, num_classes=10):
    input_tensor = keras.layers.Input(shape=shape)
    x = keras.layers.Conv2D(4, (5, 5), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    x = keras.layers.Conv2D(12, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(num_classes, name='before_softmax')(x)
    x = keras.layers.Activation('softmax', name='predictions')(x)

    return keras.models.Model(input_tensor, x)


def MLP(shape, num_classes=10):
    input_tensor = keras.layers.Input(shape=shape)
    x = keras.layers.Flatten(name='flatten')(input_tensor)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(num_classes, name='before_softmax')(x)
    x = keras.layers.Activation('softmax', name='predictions')(x)

    return keras.models.Model(input_tensor, x)

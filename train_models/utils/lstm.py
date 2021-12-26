from tensorflow import keras


def LSTM(shape):
    input_data = keras.layers.Input(shape=shape)
    output = keras.layers.LSTM(128, return_sequences=False,return_state=False)(input_data)
    output = keras.layers.Dense(128, activation='relu')(output)
    output = keras.layers.Dense(64, activation='relu')(output)
    output = keras.layers.Dense(10, activation='softmax')(output)
    model = keras.Model(inputs = input_data,outputs = output)

    return model
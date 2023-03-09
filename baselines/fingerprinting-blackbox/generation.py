import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        

def CBA(model, x, y='L', k=5, iters=1000):
    """ CBA (Classification Boundary Attack [targeted attack])
    args:
        model: victim model (logits outputs)
        x: a single input 
        y: target label selection strategy ('L' and 'R')
        k: threshold
        iters: optimization budget

    return:
        optimized input x
    """
    opt = keras.optimizers.Adam(0.001)
    relu = keras.layers.ReLU()
    z = model(x)[0].numpy()
    i = np.argmax(z) 
    
    if y == 'L':  # least-like
        j = np.argmin(z)
    elif y == 'R': # random
        ll = list(range(z.shape[0]))
        ll.remove(i)
        j = random.choice(ll)
        
    for iter in range(iters):
        z = model(x)[0].numpy()
        z[i] = -1000
        z[j] = -1000
        t = np.argmax(z) # max index except i,j
        x = tf.Variable(x)
        loss = lambda: relu(model(x)[0][i] - model(x)[0][j] + k) + relu(model(x)[0][t] - model(x)[0][i])
        step_count = opt.minimize(loss, [x]).numpy()
        if relu(model(x)[0][i] - model(x)[0][j] + k) + relu(model(x)[0][t] - model(x)[0][i]).numpy() == 0:
            return x.numpy()
        
    return x.numpy()



if __name__ == '__main__':
    dataset_name = 'CIFAR10'
    NUMS = 100  # NUMS: size of generated examples
    
    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
        
    # victim model path, replace it.     
    model_owner = load_model("./temp.h5") 
      
    # sub_model: get logits outputs. 
    sub_model = Model(inputs=model_owner.input, outputs=model_owner.layers[-2].output)
    
    advx = []
    indexs = random.sample(list(range(x_test.shape[0])), NUMS)
    for i in indexs:
        # print('Seed Index: ', i)
        x = CBA(sub_model, x_test[[i]])
        advx.append(x[0])
        
    advx = np.array(advx)
    advx_labels = keras.utils.to_categorical(np.argmax(model_owner(advx),axis=1), 10)
    
    # save the examples. 
    np.savez("./key_xy.npz", x_key=advx, y_key=advx_labels)
    print("Key (x,y) saved.")
    
    
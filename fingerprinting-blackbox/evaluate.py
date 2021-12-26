import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if __name__ == '__main__':
    # keys_path: key samples saved path 
    keys_path = './key_xy.npz'
    
    # suspect_path: suspect model path, replace it.
    suspect_path = './temp.h5'
    
    with np.load(keys_path) as f:
        x_key = f['x_key']
        y_key = f['y_key']

    suspect_model = load_model(suspect_path)
    loss, MR = suspect_model.evaluate(x_key, y_key, verbose=0)
    print('MR: ', MR)
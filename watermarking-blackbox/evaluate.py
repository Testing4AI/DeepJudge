import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
        
if __name__ == '__main__':
    # cifar10 = tf.keras.datasets.cifar10
    # (training_images, training_labels), (test_images, test_labels) = cifar10.load_data()
    # training_images = training_images / 255.0
    # test_images = test_images / 255.0
    # training_labels = tf.keras.utils.to_categorical(training_labels, 10)
    # test_labels = tf.keras.utils.to_categorical(test_labels, 10)
    
    with np.load('./logs/content_trigger.npz') as f:
        test_sample_images = f['test_sample_images']
        test_sample_labels = f['test_sample_labels']
    
    watermarked_model = load_model('./logs/watermarked_model.h5')
    # loss, acc = watermarked_model.evaluate(test_images, test_labels)
    loss, TSA = watermarked_model.evaluate(test_sample_images, test_sample_labels)
    
    # print('ACC: ', acc)
    print('TSA: ', TSA)
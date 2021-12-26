import tensorflow as tf
import numpy as np
import utils
import os
from tensorflow.keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        
def BER(model, w, b, layer_idx=None, layer_name=None):
    """calculating the bit error rate (BER)

    args:
        model: the model for the verification
        w: key matrix
        b: signature
        layer_idx: watermarked layer index
        layer_name: watermarked layer name
    
    return:
        BER
    """
    if layer_idx is None and layer_name is None:
        return -1
    if layer_idx is not None:
        layer = model.layers[layer_idx]
    if layer_name is not None:
        layer = model.get_layer(layer_name)
    x = layer.weights[0]
    y = tf.reduce_mean(x, axis=3)
    z = tf.reshape(y, (1, -1))
    _b = np.int32((tf.matmul(z, tf.cast(w, tf.float32))).numpy() >= 0)
    BER = np.sum(b != _b) / np.float(embed_dim)
    return BER      



if __name__ == '__main__':
    # load testing data
    dataset_name = 'CIFAR10'
    dataset = getattr(utils, dataset_name)
    training_images, training_labels, test_images, test_labels = dataset.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    
    embed_dim = 128
    # load the key matrix and the embedded signature b
    w = np.load('./logs/w.npy')
    b = np.load('./logs/b.npy')

    watermarked_model = load_model('./logs/watermarked_model.h5')
    watermarked_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    loss, acc = watermarked_model.evaluate(test_images, test_labels, verbose=0)
    print('ACC: ', acc)
    print('BER: ', BER(watermarked_model, w, b, layer_name='watermark_layer'))
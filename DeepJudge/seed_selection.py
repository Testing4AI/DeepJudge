import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import numpy as np
import os


parser = argparse.ArgumentParser(description='DeepJudge Seed Selection Process')
parser.add_argument('--model', required=True, type=str, help='victim model path')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset for the seed selection')
parser.add_argument('--num', default=1000, type=int, help='number of selected seeds')
parser.add_argument('--order', default='max', type=str, help='largest certainties or least. choice: max/min')
parser.add_argument('--output', default='./seeds', type=str, help='seeds saved dir')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def seedSelection(model, x, y, num=1000, order='max'):
    true_idx = np.where(np.argmax(model(x), axis=1) == np.argmax(y, axis=1))[0]
    x, y = x[true_idx], y[true_idx]
    ginis = np.sum(np.square(model(x).numpy()), axis=1)
    if order == 'max':
        ranks = np.argsort(-ginis)
    else:
        ranks = np.argsort(ginis)
    return x[ranks[:num]], y[ranks[:num]]



if __name__ == '__main__':
    opt = parser.parse_args()
    
    if opt.dataset == 'cifar10':
        cifar10 = tf.keras.datasets.cifar10
        (training_images, training_labels), (test_images, test_labels) = cifar10.load_data()
    elif opt.dataset == 'mnist':
        mnist = tf.keras.datasets.mnist
        (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
        test_images = test_images.reshape(10000, 28, 28, 1)
    else:
        raise NotImplementedError()
    
    # select seeds from the testing dataset
    x_test = test_images / 255.0
    y_test = tf.keras.utils.to_categorical(test_labels, 10)

    victim_model = load_model(opt.model)
    seeds_x, seeds_y = seedSelection(victim_model, x_test, y_test, num=opt.num, order=opt.order)
    
    log_dir = opt.output
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    save_path = f"{log_dir}/{opt.dataset}_{opt.order}_{opt.num}seeds.npz"
    np.savez(save_path, seeds_x=seeds_x, seeds_y=seeds_y)
    print('Selected seeds saved at ' + save_path)



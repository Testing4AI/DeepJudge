import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img, ImageDataGenerator
import gzip
from skimage.util.noise import random_noise
from resnet20 import resnet_v1
from lenet5 import Lenet5

os.environ["CUDA_VISIBLE_DEVICES"]='0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def log(content):
    if log_dir is not None:
        log_file = log_dir + '/log.txt'
        with open(log_file, 'a') as f:
            print(content, file=f)

        
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 70:
        lr *= 1e-3
    if epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1 
    print('Learning rate: ', lr)
    return lr


def load_data(dataset: str):
    if dataset == 'MNIST':
        mnist = tf.keras.datasets.mnist
        (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
        training_images = training_images.reshape(60000, 28, 28, 1)
        test_images = test_images.reshape(10000, 28, 28, 1)
    elif dataset == 'CIFAR10':
        cifar10 = tf.keras.datasets.cifar10
        (training_images, training_labels), (test_images, test_labels) = cifar10.load_data()

    return training_images, training_labels, test_images, test_labels


def get_unrelated_images(dataset: str, sample_rate):
    watermark_images = []
    if dataset == 'MNIST':
        # e-mnist for the mnist dataset 
        train_images_path = './data/emnist/emnist-letters-train-images-idx3-ubyte.gz'
        train_labels_path = './data/emnist/emnist-letters-train-labels-idx1-ubyte.gz'
        with gzip.open(train_images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape((-1, 28, 28, 1))
        with gzip.open(train_labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        for i in range(images.shape[0]):
            if labels[i] == 23:
                watermark_images.append(images[i])
    elif dataset == 'CIFAR10':
        # mnist for the cifar10 dataset 
        mnist = tf.keras.datasets.mnist
        (training_images, training_labels), (_, _) = mnist.load_data()
        for i in range(len(training_labels)):
            if training_labels[i] == 1:
                image = array_to_img(training_images[i].reshape(28, 28, 1))
                image = image.convert(mode='RGB')
                image = image.resize((32, 32))
                image = img_to_array(image)
                watermark_images.append(image)

    random.shuffle(watermark_images)
    watermark_images = np.array(watermark_images)
    train_sample_number = int(len(watermark_images) * sample_rate)
    train_sample = watermark_images[:train_sample_number]
    test_sample = watermark_images[train_sample_number:]

    return train_sample, test_sample


def watermark(train_images, train_labels, old_label, new_label, sample_rate, dataset: str, wtype='content'):
    """prepare the dataset for training to embed the watermark 
    
    args:
        train_images: clean training images
        train_labels: clean training labels
        old_label: label for watermarking
        new_label: label after watermarking
        sample_rate: sample rate for embedding the watermark
        wtype: watermarking type ('content', 'noise', 'unrelated')
    
    return:
        processed training and testing dataset for watermarking
    """
    if wtype == 'unrelated':
        train_sample, test_sample = get_unrelated_images(dataset, sample_rate)
    else:
        watermark_images = []
        for i in range(len(train_labels)):
            if train_labels[i] == old_label:
                watermark_images.append(train_images[i])
                
        if wtype == 'content':
            # add the trigger (size= 8*8) at the right bottom corner 
            mark_image = load_img('./mark/apple_black.png', color_mode='grayscale', target_size=(8, 8))
            for i in range(len(watermark_images)):
                image = array_to_img(watermark_images[i])
                image.paste(mark_image, box=(image.size[0] - 8, image.size[1] - 8))
                watermark_images[i] = img_to_array(image)
        elif wtype == 'noise':
            for i in range(len(watermark_images)):
                image = random_noise(watermark_images[i] / 255.0, seed=1)
                image = image * 255.0
                watermark_images[i] = image
                
        random.shuffle(watermark_images)
        watermark_images = np.array(watermark_images)
        train_sample_number = int(len(watermark_images) * sample_rate)
        train_sample = watermark_images[:train_sample_number]
        test_sample = watermark_images[train_sample_number:]

    if dataset == 'MNIST':
        return train_sample, np.ones(train_sample.shape[0]) * new_label, test_sample, np.ones(
            test_sample.shape[0]) * new_label
    elif dataset == 'CIFAR10':
        return train_sample, np.ones((train_sample.shape[0], 1)) * new_label, test_sample, np.ones((
            test_sample.shape[0], 1)) * new_label



if __name__ == '__main__':    
    wtype = 'content'
    dataset = 'CIFAR10'
    training_nums = 25000 
    batch_size = 64
    epochs = 80  # 80 for cifar10 and 10 for mnist
    no_augmentation = False
    old_label = 1
    new_label = 3
    log_dir = './logs' 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print('log saved at ' + log_dir)
    
    cifar10 = tf.keras.datasets.cifar10
    (training_images, training_labels), (test_images, test_labels) = cifar10.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    training_labels = tf.keras.utils.to_categorical(training_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)
    
    training_images, training_labels, test_images, test_labels = load_data(dataset)
    train_sample_images, train_sample_labels, test_sample_images, test_sample_labels = watermark(training_images,
                                                                                                 training_labels, old_label, new_label,
                                                                                                 0.1, dataset,
                                                                                                 wtype=wtype)

    training_labels = tf.keras.utils.to_categorical(training_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)
    train_sample_labels = tf.keras.utils.to_categorical(train_sample_labels, 10)
    test_sample_labels = tf.keras.utils.to_categorical(test_sample_labels, 10)

    training_images = training_images / 255.0
    test_images = test_images / 255.0
    train_sample_images = train_sample_images / 255.0
    test_sample_images = test_sample_images / 255.0
    training_all_images = np.concatenate((training_images[:training_nums], train_sample_images), axis=0)
    training_all_labels = np.concatenate((training_labels[:training_nums], train_sample_labels), axis=0)
    
    input_shape = training_images.shape[1:]
    model = resnet_v1(input_shape=input_shape, depth=20) 
    model.summary(print_fn=log)
    
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                      cooldown=0,
                                                      patience=5,
                                                      min_lr=0.5e-6)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if no_augmentation:
        print('Not using data augmentation.')
        history = model.fit(training_all_images, training_all_labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(test_images, test_labels),
                            callbacks=[reduce_lr, lr_reducer])
    else:
        print('Using real-time data augmentation.')
        data_gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        history = model.fit(data_gen.flow(training_all_images, training_all_labels, batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(test_images, test_labels),
                            callbacks=[reduce_lr, lr_reducer],
                            steps_per_epoch=training_all_images.shape[0] // batch_size)

    pd.DataFrame(history.history).to_csv(log_dir + '/log.csv')
    
    if log_dir is not None:
        model.save(log_dir + '/watermarked_model.h5')
        np.savez(log_dir + "/content_trigger.npz", test_sample_images=test_sample_images, test_sample_labels=test_sample_labels)
        
#     loss, TSA = model.evaluate(test_sample_images, test_sample_labels)
    
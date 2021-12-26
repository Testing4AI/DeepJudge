import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import utils
from regularization import WatermarkRegularizer


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    # embedding parameters
    dataset_name = 'CIFAR10'
    dataset = getattr(utils, dataset_name)
    training_nums = 25000
    batch_size = 64
    epochs = 80
    no_augmentation = False
    log_dir = "./logs" 
    scale = 0.01
    embed_dim = 128
    wtype = 'random'
    embed_dense = 2 # conv2 block
    no_embed = False

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print('log saved at ' + log_dir)
    log('dataset: ' + dataset_name)
    log('training_nums: ' + str(training_nums))
    log('batch_size: ' + str(batch_size))
    log('epochs: ' + str(epochs))
    log('no_augmentation: ' + str(no_augmentation))
    log('scale: ' + str(scale))
    log('embed_dim: ' + str(embed_dim))
    log('wtype: ' + wtype)
    log('embed_dense: ' + str(embed_dense))

    # load data
    training_images, training_labels, test_images, test_labels = dataset.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    # embed the watermark into model
    b = np.random.randint(0, 2, (1, embed_dim))
    watermark = WatermarkRegularizer(scale, b, embed_dense, no_embed=no_embed, wtype=wtype)

    model = dataset.get_model(training_images.shape[1:], watermark)
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
        print('White-box watermark embedding: Not using data augmentation.')
        history = model.fit(training_images[:training_nums], training_labels[:training_nums],
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(test_images, test_labels),
                            callbacks=[reduce_lr, lr_reducer])
    else:
        print('White-box watermark embedding: Using real-time data augmentation.')
        data_gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        history = model.fit(
            data_gen.flow(training_images[:training_nums], training_labels[:training_nums], batch_size=batch_size),
            epochs=epochs,
            validation_data=(test_images, test_labels),
            callbacks=[reduce_lr, lr_reducer],
            steps_per_epoch=training_nums // batch_size)

    if log_dir is not None:
        pd.DataFrame(history.history).to_csv(log_dir + '/log.csv')

    # save w and b
    watermarked_model = dataset.get_validate_model(training_images.shape[1:], model, watermark)
    if watermark.w is not None and log_dir is not None:
        watermarked_model.save(log_dir + '/watermarked_model.h5')
        np.save(log_dir + "/w.npy", watermark.w)
        np.save(log_dir + "/b.npy", b)
        
#     watermarked_model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])        
#     loss, acc = watermarked_model.evaluate(test_images, test_labels)
#     log('ACC: ' + acc)
#     ber = BER(watermarked_model, w, b, layer_name='watermark_layer')
#     log('BER: ' + str(ber))











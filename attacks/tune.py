import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_model_optimization as tfmot



# Default data-augmentation parameters
RANGE = 10
W_SHIFT = 0.1
H_SHIFT = 0.1


def FTLL(model, x, y, epochs=10, lr=None, batch_size=64, aug=True):
    """ Fine-tuning only the last layer
    
    args:
        model: victim model
        x: subset of data for fine-tuning
        y: ground-truth labels
        epochs: fine-tuning epochs
        lr: fine-tuning learning rate   
        batch_size: fine-tuning batch-size
        aug: whether using data-augmentation during fine-tuning
    
    return:
        fine-tuned model
    """

    for layer in model.layers[:-2]:
        layer.trainable = False
    
    if lr is not None:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])
        
    if aug:
        print('FTLL: Using real-time data augmentation.')
        data_gen = ImageDataGenerator(rotation_range=RANGE, width_shift_range=W_SHIFT, height_shift_range=H_SHIFT)
        data_gen.fit(x)
        model.fit_generator(data_gen.flow(x, y, batch_size=batch_size), epochs=epochs, steps_per_epoch=x.shape[0]//batch_size, verbose=0)
    else:
        print('FTLL: Not using data augmentation.')
        model.fit(x, y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0)
        
    print("Done")
    return model
    
    
def FTAL(model, x, y, epochs=10, lr=None, batch_size=64, aug=True):
    """ Fine-tuning all the layers
    
    args:
        model: victim model
        x: subset of data for fine-tuning
        y: ground-truth labels
        epochs: fine-tuning epochs
        lr: fine-tuning learning rate    
        batch_size: fine-tuning batch-size
        aug: whether using data-augmentation during fine-tuning
    
    return:
        fine-tuned model
    """

    if lr is not None:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])
        
    if aug:
        print('FTAL: Using real-time data augmentation.')
        data_gen = ImageDataGenerator(rotation_range=RANGE, width_shift_range=W_SHIFT, height_shift_range=H_SHIFT)
        data_gen.fit(x)
        model.fit_generator(data_gen.flow(x, y, batch_size=batch_size), epochs=epochs, steps_per_epoch=x.shape[0]//batch_size, verbose=0)
    else:
        print('FTAL: Not using data augmentation.')
        model.fit(x, y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0)
        
    print("Done")    
    return model


def RTAL(model, x, y, epochs=10, lr=None, batch_size=64, aug=True):
    """ Re-initializing the last layer and fine-tuning all the layers
    
    args:
        model: victim model
        x: subset of data for fine-tuning
        y: ground-truth labels
        epochs: fine-tuning epochs
        lr: fine-tuning learning rate    
        batch_size: fine-tuning batch-size
        aug: whether using data-augmentation during fine-tuning
    
    return:
        fine-tuned model
    """

    if lr is None:
        lr = model.optimizer.lr 
        
    num_classes = model.output.shape[1]
    sub_model = Model(inputs=model.input, outputs=model.layers[-3].output)
    out = Dense(num_classes, kernel_initializer='he_normal', name='output_dense')(sub_model.output)
    outs = Activation('softmax', name='output_activation')(out)
    model = Model(inputs=sub_model.input, outputs=outs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    
    if aug:
        print('RTAL: Using real-time data augmentation.')
        data_gen = ImageDataGenerator(rotation_range=RANGE, width_shift_range=W_SHIFT, height_shift_range=H_SHIFT)
        data_gen.fit(x)
        model.fit_generator(data_gen.flow(x, y, batch_size=batch_size), epochs=epochs, steps_per_epoch=x.shape[0]//batch_size, verbose=0)
    else:
        print('RTAL: Not using data augmentation.')
        model.fit(x, y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0)
        
    print("Done")    
    return model


def Prune(model, x, y, r=0.2, epochs=10, lr=None, batch_size=64, aug=True):
    """ Weight pruning 
    
    args:
        model: victim model
        x: subset of data for fine-tuning
        y: ground-truth labels
        r: pruning rate (0.2 means pruning 20%)
        epochs: fine-tuning epochs
        lr: fine-tuning learning rate   
        batch_size: fine-tuning batch-size
        aug: whether using data-augmentation during fine-tuning
    
    return:
        pruned model
    """

    if lr is None:
        lr = model.optimizer.lr 
        
    end_step = np.ceil(1.0 * x.shape[0] / batch_size).astype(np.int32) * epochs
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    PolynomialDecay = tfmot.sparsity.keras.PolynomialDecay
    new_pruning_params = {
        'pruning_schedule': PolynomialDecay(initial_sparsity=0.0,
                                            final_sparsity=r,
                                            begin_step=0,
                                            end_step=end_step)  
    }
    new_pruned_model = prune_low_magnitude(model, **new_pruning_params)
    new_pruned_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                             loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    
    if aug:
        print('Prune: Using real-time data augmentation.')
        data_gen = ImageDataGenerator(rotation_range=RANGE, width_shift_range=W_SHIFT, height_shift_range=H_SHIFT)
        new_pruned_model.fit(data_gen.flow(x, y, batch_size=batch_size), epochs=epochs, 
                             steps_per_epoch=x.shape[0]//batch_size, callbacks=callbacks, verbose=0)
    else:
        print('Prune: No data augmentation.')
        new_pruned_model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)
        
    model = tfmot.sparsity.keras.strip_pruning(new_pruned_model)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                        loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Done")
    return model
        
        

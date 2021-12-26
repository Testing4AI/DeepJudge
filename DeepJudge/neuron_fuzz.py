from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf



class NeuronFuzzing:
    def __init__(self, model):
        """ Generate white-box test cases for neurons
        args:
            model: victim model 
        """
        self.model = model 
        
    def getKs(submodel, x, times):
        """ calculate threshold k for each neuron
        """
        outputs = submodel(x)
        shape = outputs.shape
        outputs = K.mean(K.reshape(outputs, (shape[0], -1, shape[-1])), axis = 1)
        layer_maxs = np.max(outputs, axis=0)
        return (times * layer_maxs)

    def generate(self, seeds, layer_index, m=3, iters=1000, step=0.01, target_idx=None, X=None):
        """ 
        args:
            seeds: seeds for the generation
            layer_index: target layer index 
            m: hyper-parameter
            iters: iteration budget
            step: optimization step
            target_idx: target neuron (optional)
            X: training data (optional)

        return:
            a dictionary of generated test cases {(layer_index, neuron_index): [test cases...]}
        """
        submodel = Model(inputs = self.model.input, outputs = self.model.layers[layer_index].output)
        output_shape = submodel.output.shape

        if X is None:
            Ks = NeuronFuzzing.getKs(submodel, seeds, m)
        else: 
            Ks = NeuronFuzzing.getKs(submodel, X, m)

        if target_idx is None:
            target_idxs = list(range(output_shape[-1]))   
        else:
            target_idxs = [target_idx]
            
        tests = {}
        for idx in target_idxs: 
            tests[(layer_index, idx)] = []
        
        for i in range(seeds.shape[0]):
            x = seeds[[i]]
            if (i+1) % 10 == 0:
                print(f"{i+1} seeds processed.")
                
            for idx in target_idxs:
                opt = keras.optimizers.Adam(step)
                x_ = tf.Variable(x)
                k = Ks[idx]
                for iter in range(iters):
                    loss = lambda: -K.mean(K.reshape(submodel(x_), (1, -1, output_shape[-1])), axis=1)[0][idx]
                    step_count = opt.minimize(loss, [x_]).numpy()
                    if K.mean(K.reshape(submodel(x_), (1, -1, output_shape[-1])), axis=1)[0][idx] > k:
                        tests[(layer_index, idx)].append(x_.numpy().reshape(self.model.input.shape[1:]))
                        break 
            
        for idx in target_idxs: 
            tests[(layer_index, idx)] = np.array(tests[(layer_index, idx)])
            
        return tests

    

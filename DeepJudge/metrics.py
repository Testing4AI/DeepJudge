import numpy as np
import scipy.stats
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K



DIGISTS = 4

def Rob(model, advx, advy):
    """ Robustness (empirical) 
    args:
        model: suspect model
        advx: black-box test cases (adversarial examples) 
        advy: ground-truth labels
    
    return:
        Rob value
    """
    return round(np.sum(np.argmax(model.predict(advx), axis=1)==np.argmax(advy, axis=1))/advy.shape[0], DIGISTS)


def JSD(model1, model2, advx):
    """ Jensen-Shanon Distance
    args:
        model1 & model2: victim model and suspect model
        advx: black-box test cases 

    return:
        JSD value
    """
    vectors1 = model1.predict(advx)
    vectors2 = model2.predict(advx)
    mid = (vectors1 + vectors2)/2
    distances = (scipy.stats.entropy(vectors1, mid, axis=1) + scipy.stats.entropy(vectors2, mid, axis=1))/2
    return round(np.average(distances), DIGISTS)


def LOD(model1, model2, tests, order=2):
    """ Layer Outputs Distance
    args:
        model1 & model2: victim model and suspect model
        tests: white-box test cases  
        order: distance norm 

    return:
        LOD value 
    """
    lods = []
    for loc in tests.keys():
        layer_index, idx = loc[0], loc[1]
        submodel1 = Model(inputs = model1.input, outputs = model1.layers[layer_index].output)
        submodel2 = Model(inputs = model2.input, outputs = model2.layers[layer_index].output)
        outputs1 = submodel1(tests[loc])
        outputs1 = K.mean(K.reshape(outputs1, (outputs1.shape[0], -1, outputs1.shape[-1])), axis = 1)
        outputs2 = submodel2(tests[loc])
        outputs2 = K.mean(K.reshape(outputs2, (outputs2.shape[0], -1, outputs2.shape[-1])), axis = 1)
        lods.append(np.linalg.norm(outputs1 - outputs2, axis=1, ord=order))
    return round(np.average(np.array(lods)), DIGISTS)


def LAD(model1, model2, tests, theta=0.5):
    """ Layer Activation Distance
    args:
        model1 & model2: victim model and suspect model
        tests: white-box test cases  
        theta: activation threshold 

    return:
        LAD value 
    """
    def normalize(vs):
        return [(v-np.min(v))/(np.max(v)-np.min(v)+1e-6) for v in vs]
    
    lads = []
    for loc in tests.keys():
        layer_index, idx = loc[0], loc[1]
        submodel1 = Model(inputs = model1.input, outputs = model1.layers[layer_index].output)
        submodel2 = Model(inputs = model2.input, outputs = model2.layers[layer_index].output)
        outputs1 = submodel1(tests[loc])
        outputs1 = K.mean(K.reshape(outputs1, (outputs1.shape[0], -1, outputs1.shape[-1])), axis = 1)
        outputs2 = submodel2(tests[loc])
        outputs2 = K.mean(K.reshape(outputs2, (outputs2.shape[0], -1, outputs2.shape[-1])), axis = 1)
        outputs1_normlized = normalize(outputs1)
        outputs2_normlized = normalize(outputs2)
        activations1 = np.array([np.where(i>theta, 1, 0) for i in outputs1_normlized])
        activations2 = np.array([np.where(i>theta, 1, 0) for i in outputs2_normlized])
        lads.append(np.linalg.norm(activations1 - activations2, axis=1, ord=1))
    return round(np.average(np.array(lads)), DIGISTS)


def NOD(model1, model2, tests):
    """ Neuron Output Distance
    args:
        model1 & model2: victim model and suspect model
        tests: white-box test cases  

    return:
        NOD value 
    """
    nods = []
    for loc in tests.keys():
        layer_index, idx = loc[0], loc[1]
        submodel1 = Model(inputs = model1.input, outputs = model1.layers[layer_index].output)
        submodel2 = Model(inputs = model2.input, outputs = model2.layers[layer_index].output)
        outputs1 = submodel1(tests[loc])
        outputs1 = K.mean(K.reshape(outputs1, (outputs1.shape[0], -1, outputs1.shape[-1])), axis = 1)
        outputs2 = submodel2(tests[loc])
        outputs2 = K.mean(K.reshape(outputs2, (outputs2.shape[0], -1, outputs2.shape[-1])), axis = 1)
        nods.append(np.abs(outputs1[:,idx] - outputs2[:,idx]))
    return round(np.average(np.array(nods)), DIGISTS)


def NAD(model1, model2, tests, theta=0.5):
    """ Neuron Activation Distance
    args:
        model1 & model2: victim model and suspect model
        tests: white-box test cases  
        theta: activation threshold 

    return:
        NAD value 
    """
    def normalize(vs):
        return [(v-np.min(v))/(np.max(v)-np.min(v)+1e-6) for v in vs]
    
    nads = []
    for loc in tests.keys():
        layer_index, idx = loc[0], loc[1]
        submodel1 = Model(inputs = model1.input, outputs = model1.layers[layer_index].output)
        submodel2 = Model(inputs = model2.input, outputs = model2.layers[layer_index].output)
        outputs1 = submodel1(tests[loc])
        outputs1 = K.mean(K.reshape(outputs1, (outputs1.shape[0], -1, outputs1.shape[-1])), axis = 1)
        outputs2 = submodel2(tests[loc])
        outputs2 = K.mean(K.reshape(outputs2, (outputs2.shape[0], -1, outputs2.shape[-1])), axis = 1)
        outputs1_normlized = normalize(outputs1)
        outputs2_normlized = normalize(outputs2)
        activations1 = np.array([np.where(i>theta, 1, 0) for i in outputs1_normlized])
        activations2 = np.array([np.where(i>theta, 1, 0) for i in outputs2_normlized])
        nads.append(np.abs(activations1[:,idx] - activations2[:,idx]))
    return round(np.average(np.array(nads))*len(tests), DIGISTS)



def NNOD(model1, model2, tests):
    """ Normalized Neuron Output Distance
    args:
        model1 & model2: victim model and suspect model
        tests: white-box test cases  

    return:
        NNOD value 
    """
    nnods = []
    for loc in tests.keys():
        layer_index, idx = loc[0], loc[1]
        submodel1 = Model(inputs = model1.input, outputs = model1.layers[layer_index].output)
        submodel2 = Model(inputs = model2.input, outputs = model2.layers[layer_index].output)
        outputs1 = submodel1(tests[loc])
        outputs1 = K.mean(K.reshape(outputs1, (outputs1.shape[0], -1, outputs1.shape[-1])), axis = 1)
        outputs2 = submodel2(tests[loc])
        outputs2 = K.mean(K.reshape(outputs2, (outputs2.shape[0], -1, outputs2.shape[-1])), axis = 1)
        maxs1 = np.max(outputs1, axis=0)
        maxs2 = np.max(outputs2, axis=0)
        outputs1 = outputs1/maxs1
        outputs2 = outputs2/maxs2
        nnods.append(np.abs(outputs1[:,idx] - outputs2[:,idx]))
    return round(np.average(np.array(nnods)), DIGISTS)



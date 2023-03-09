# Extension Modules to defend against more threats

# def NeuronMatching(suspect_model, tests):


def NNOD_Matrix(model1, model2, tests):
    X = []
    for loc in tests.keys():
        layer_index, idx = loc[0], loc[1]
        X.append(tests[loc])
    X = np.concatenate(np.array(X),axis=0)
    
    submodel1 = Model(inputs = model1.input, outputs = model1.layers[layer_index].output)
    submodel2 = Model(inputs = model2.input, outputs = model2.layers[layer_index].output)
    outputs1 = submodel1(X)
    outputs1 = K.mean(K.reshape(outputs1, (outputs1.shape[0], -1, outputs1.shape[-1])), axis = 1)
    outputs2 = submodel2(X)
    outputs2 = K.mean(K.reshape(outputs2, (outputs2.shape[0], -1, outputs2.shape[-1])), axis = 1)
    
    maxs1 = np.max(outputs1, axis=0)
    maxs2 = np.max(outputs2, axis=0)
    
    outputs1 = outputs1/maxs1
    outputs2 = outputs2/maxs2
    
    NOD_matrix = [[] for _ in range(len(tests.keys()))]
    for idx in range(len(tests.keys())):
        for j in range(len(tests.keys())):
            NOD_matrix[idx].append(round(np.average(np.abs(outputs1[:,idx]-outputs2[:,j])), 4))
    return np.array(NOD_matrix)


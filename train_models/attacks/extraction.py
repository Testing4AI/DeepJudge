import tensorflow as tf
import numpy as np


class BlackboxModel:
    def __init__(self, victim_model):
        self.model = victim_model
    
    def query(self, X):
        return self.model(X).numpy() 
    

class KnockoffClient:
    def __init__(self, x_sub, size=None, train_epoch=10, batch_size=64, lr=0.001):
        """
        args:
            x_sub: auxiliary dataset
            size: random sampling size (0.5 for 50%, default is 100%)
            train_epoch: training epochs 
            batch_size: training batch-size
            lr: training learning rate
        """
        self.x_sub = x_sub
        self.size = size
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.lr = lr
        
    def extract(self, substitute_model, blackbox_model):
        """
        args:
            substitute_model: null model
            blackbox_model: victim model API
        
        return:
            extracted mdoel
        """
        substitute_model.compile(optimizer=tf.keras.optimizers.Adam(self.lr),
                          loss='categorical_crossentropy', metrics=['accuracy'])    
        if self.size is None:
            x_subset = self.x_sub 
        else:
            sample_size = int(self.x_sub.shape[0]*self.size)
            idx = np.random.choice(self.x_sub.shape[0], sample_size, replace=False)
            x_subset = self.x_sub[idx]
        print(f"    Random sampling size: {x_subset.shape[0]}")
        victim_outputs = blackbox_model.query(x_subset)
        substitute_model.fit(x_subset, victim_outputs, epochs=self.train_epoch, 
                             batch_size=self.batch_size, verbose=0)
        
        return substitute_model

    

class JBAClient:
    def __init__(self, x_seeds, extract_round=6, tau=2, scale_const=0.01, train_epoch=10, batch_size=64, lr=0.001):
        """
        args:
            x_seeds: seed inputs for the augmentation
            extract_round: the round for the augmentation
            scale_const: optimization step
            train_epoch: training epochs 
            batch_size: training batch-size
            lr: training learning rate
        """
        self.x_seeds = x_seeds
        self.extract_round = extract_round
        self.tau = tau
        self.scale_const = scale_const
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.lr = lr
        
    def extract(self, substitute_model, blackbox_model):
        """
        args:
            substitute_model: null model
            blackbox_model: victim model API
        
        return:
            extracted mdoel
        """
        substitute_model.compile(optimizer=tf.keras.optimizers.Adam(self.lr),
                          loss='categorical_crossentropy', metrics=['accuracy'])  
        x_substitute = self.x_seeds
        print(f"    Number of seeds: {x_substitute.shape[0]}")
        victim_outputs = blackbox_model.query(x_substitute)
        for i in range(1, self.extract_round+1): 
            scale = self.scale_const*(-1)**(int(i/self.tau))
            substitute_model.fit(x_substitute, victim_outputs, batch_size=self.batch_size,
                                 epochs=self.train_epoch, verbose=0)
            if i != self.extract_round:
                x_new = x_substitute
                for j in range(0, len(x_new), 500):
                    # query batch-size is 500
                    x_slice = x_new[j:j+500]
                    x_slice = tf.Variable(x_slice)
                    with tf.GradientTape() as tape:
                        tape.watch(x_slice)
                        outputs = substitute_model(x_slice)
                        grads = tape.gradient(outputs, x_slice)
                    delta = tf.sign(grads)
                    x_slice.assign_add(scale * delta)
                    x_slice = x_slice.numpy()
                    y_new = blackbox_model.query(x_slice)
                    x_substitute = np.concatenate((x_substitute, x_slice), axis=0)
                    victim_outputs = np.concatenate((victim_outputs, y_new), axis=0)

        return substitute_model
    
    
class SyntheticGenerator:
    def __init__(self, model, step=0.01, epochs=30):
        self.model = model
        self.step = step
        self.epochs = epochs
    
    def generate(self, x, y):
        opt = tf.keras.optimizers.Adam(learning_rate=self.step)
        x_sys = tf.Variable(x)
        target = tf.constant(y)
        for j in range(self.epochs):
            loss = lambda: tf.keras.losses.categorical_crossentropy(target, self.model(x_sys))
            step_count = opt.minimize(loss, [x_sys]).numpy()
            x_sys = tf.Variable(x_sys)
        return x_sys.numpy()
    
    
class ESAClient:
    def __init__(self, extract_epoch, syns_num, syns_epoch, syns_step, train_epoch=10, batch_size=64, lr=0.001):
        """
        args:
            extract_epoch: extracting epoch
            syns_num: the size of synthesis inputs 
            syns_epoch: synthesis epochs
            syns_step: synthesis step
            train_epoch: training epochs 
            batch_size: training batch-size
            lr: training learning rate
        """
        self.extract_epoch = extract_epoch
        self.syns_num = syns_num
        self.syns_epoch = syns_epoch
        self.syns_step = syns_step
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.lr = lr
        
    def extract(self, substitute_model, blackbox_model):
        """
        args:
            substitute_model: null model
            blackbox_model: victim model API
        
        return:
            extracted mdoel
        """
        substitute_model.compile(optimizer=tf.keras.optimizers.Adam(self.lr),
                          loss='categorical_crossentropy', metrics=['accuracy']) 
        num_classes = substitute_model.output.shape[-1]
        size = tuple([self.syns_num]+list(substitute_model.input.shape[1:]))
        synthesis_x = np.random.random(size)
        print(f"    Total Extraction Epochs: {self.extract_epoch}")
        print(f"    Synthesis X size: {size}")
        for i in range(1, self.extract_epoch+1):
#             print("    Extration Epoch {}".format(i))
            victim_outputs = blackbox_model.query(synthesis_x)
            substitute_model.fit(synthesis_x, victim_outputs, epochs=self.train_epoch, 
                                 batch_size=self.batch_size, verbose=0)
            if i != self.extract_epoch: 
                synthesis_x = np.random.random(size=size)
                synthesis_y = []
                for j in range(self.syns_num):
                    alpha = np.random.randint(1, 1000, size=num_classes)
                    synthesis_y.append(np.random.dirichlet(alpha))
                synthesis_y = np.array(synthesis_y)
                synthetic_generator = SyntheticGenerator(substitute_model, step=self.syns_step, epochs=self.syns_epoch)
                synthesis_x = synthetic_generator.generate(synthesis_x, synthesis_y)
        
        return substitute_model




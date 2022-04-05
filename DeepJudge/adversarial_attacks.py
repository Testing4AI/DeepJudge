from tensorflow import keras
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.models import Model



class FGSM:
    def __init__(self, model, ep=0.3, isRand=True, clip_min=0, clip_max=1):
        """
        args:
            model: victim model
            ep: FGSM perturbation bound 
            isRand: whether adding a random noise 
            clip_min: clip lower bound
            clip_max: clip upper bound
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        self.clip_min = clip_min
        self.clip_max = clip_max
        
    def generate(self, x, y, randRate=1):
        """ 
        args:
            x: normal inputs 
            y: ground-truth labels
            randRate: the size of random noise 

        returns:
            successed adversarial examples and corresponding ground-truth labels
        """
        target = tf.constant(y)
        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, self.clip_min, self.clip_max)
        
        x_adv = tf.Variable(x)
        with tf.GradientTape() as tape:
            loss = keras.losses.categorical_crossentropy(target, self.model(x_adv))
            grads = tape.gradient(loss, x_adv)
        delta = tf.sign(grads)
        x_adv.assign_add(self.ep * delta)
        x_adv = tf.clip_by_value(x_adv, clip_value_min=self.clip_min, clip_value_max=self.clip_max)
        
        success_idx = np.where(np.argmax(self.model(x_adv), axis=1) != np.argmax(y, axis=1))[0] 
        print("SUCCESS:", len(success_idx))
        return x_adv.numpy()[success_idx], y[success_idx]


class PGD: 
    def __init__(self, model, ep=0.3, epochs=10, step=0.03, isRand=True, clip_min=0, clip_max=1):
        """
        args:
            model: victim model
            ep: PGD perturbation bound 
            epochs: PGD iterations 
            isRand: whether adding a random noise 
            clip_min: clip lower bound
            clip_max: clip upper bound
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        self.epochs = epochs
        self.step = step
        self.clip_min = clip_min
        self.clip_max = clip_max
        
    def generate(self, x, y, randRate=1):
        """ 
        args:
            x: normal inputs 
            y: ground-truth labels
            randRate: the size of random noise 

        returns:
            successed adversarial examples and corresponding ground-truth labels
        """
        target = tf.constant(y)
        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, self.clip_min, self.clip_max)
        
        x_adv = tf.Variable(x)
        for i in range(self.epochs): 
            with tf.GradientTape() as tape:
                loss = keras.losses.categorical_crossentropy(target, self.model(x_adv))
                grads = tape.gradient(loss, x_adv)
            delta = tf.sign(grads)
            x_adv.assign_add(self.step * delta)
            x_adv = tf.clip_by_value(x_adv, clip_value_min=self.clip_min, clip_value_max=self.clip_max)
            x_adv = tf.Variable(x_adv)
            
        success_idx = np.where(np.argmax(self.model(x_adv), axis=1) != np.argmax(y, axis=1))[0] 
        print("SUCCESS:", len(success_idx))
        return x_adv.numpy()[success_idx], y[success_idx]
    
    
class CW_L2:
    def __init__(self, model, batch_size, confidence, targeted,
                 learning_rate, binary_search_steps, max_iterations,
                 abort_early, initial_const, clip_min, clip_max, shape):
        """ a tf2 version of C&W-L2 (batch generation)
            based on https://github.com/cleverhans-lab/cleverhans
        """
        self.TARGETED = targeted
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = model
        self.sub_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.repeat = binary_search_steps >= 10
        self.shape = tuple([batch_size] + list(shape))
        self.modifier = tf.Variable(np.zeros(self.shape, dtype=np.dtype('float32')))
    
    def ZERO():
        return np.asarray(0., dtype=np.dtype('float32'))
    
    def attack(self, images, targets):
        """
        Perform the L_2 attack on the given instance for the given targets.
        If self.targeted is true, then the targets represents the target labels
        If self.targeted is false, then targets are the original class labels
        """
        r = []
        for i in range(0, images.shape[0], self.batch_size):
            tf.print('Processing {} - {} inputs'.format(i+1, i+self.batch_size))
            r.extend(self.attack_batch(images[i:i + self.batch_size],
                                       targets[i:i + self.batch_size]))
        
        success_idx = np.where(np.argmax(self.model(np.array(r)), axis=1) != np.argmax(targets, axis=1))[0] 
        print("SUCCESS:", len(success_idx))
        return np.array(r)[success_idx], targets[success_idx]
    
    def attack_batch(self, imgs, labs):
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = tf.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y
        
        batch_size = self.batch_size
        oimgs = np.clip(imgs, self.clip_min, self.clip_max)
        imgs = (imgs - self.clip_min) / (self.clip_max - self.clip_min)
        imgs = np.clip(imgs, 0, 1)
        imgs = (imgs * 2) - 1
        imgs = np.arctanh(imgs * .999999)
        
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10
        
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = np.copy(oimgs)
        
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            tf.print("    BINARY_SEARCH_STEPS:", outer_step)
            self.modifier.assign(tf.zeros(self.shape, dtype=tf.float32))
            batch = tf.cast(imgs[:batch_size], dtype=tf.float32)
            batchlab = tf.cast(labs[:batch_size], dtype=tf.float32)
            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size
            if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound
            const = tf.cast(CONST, dtype=tf.float32)
            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                with tf.GradientTape() as tape:
                    tape.watch(self.modifier)
                    newimg = (tf.tanh(self.modifier + batch) + 1) / 2
                    newimg = newimg * (self.clip_max - self.clip_min) + self.clip_min
                    output = self.sub_model(newimg)
                    
                    other = (tf.tanh(batch) + 1) / 2 * (self.clip_max - self.clip_min) + self.clip_min
                    l2dist = tf.reduce_sum(tf.square(newimg - other), list(range(1, len(imgs.shape))))
                    real = tf.reduce_sum(batchlab * output, 1)
                    another = tf.reduce_max((1 - batchlab) * output - batchlab * 10000, 1)
                    
                    if self.TARGETED:
                        # if targeted, optimize for making the other class most likely
                        loss1 = tf.maximum(CW_L2.ZERO(), another - real + self.CONFIDENCE)
                    else:
                        # if untargeted, optimize for making this class least likely.
                        loss1 = tf.maximum(CW_L2.ZERO(), real - another + self.CONFIDENCE)
                    loss2 = tf.reduce_sum(l2dist)
                    loss1 = tf.reduce_sum(const * loss1)
                    loss = loss1 + loss2
                    grads = tape.gradient(loss, self.modifier)
                self.optimizer.apply_gradients(zip([grads], [self.modifier]))
                if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    tf.print(("    Iteration {} of {}: loss={:.3g} " +
                              "l2={:.3g} f={:.3g}").format(
                        iteration, self.MAX_ITERATIONS, loss,
                        tf.reduce_mean(l2dist), tf.reduce_mean(output)))
                if self.ABORT_EARLY and iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    if loss > prev * .9999:
                        msg = "    Failed to make progress; stop early"
                        tf.print(msg)
                        break
                    prev = loss
                
                for e, (l2, sc, ii) in enumerate(zip(l2dist, output, newimg)):
                    lab = np.argmax(batchlab[e])
                    if l2 < bestl2[e] and compare(sc, lab):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, lab):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii
            
            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10
            o_bestl2 = np.array(o_bestl2)
        return o_bestattack                 


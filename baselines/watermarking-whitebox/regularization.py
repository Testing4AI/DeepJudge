from tensorflow.keras.regularizers import Regularizer
import numpy as np
import tensorflow as tf



def random_index_generator(count):
    indices = np.arange(0, count)
    np.random.shuffle(indices)
    for idx in indices:
        yield idx


class WatermarkRegularizer(Regularizer):
    """
    b       signature 
    wtype   'random' 'direct' 'diff'
    w       key matrix 
    """
    def __init__(self, scale, b, target_block=1, no_embed=False, wtype='random', w=None, rand_seed='none'):
        self.scale = float(scale)
        self.b = b
        self.wtype = wtype
        self.w = w
        self.target_block = target_block
        self.no_embed = no_embed
        if rand_seed == 'time':
            import time
            np.random.seed(int(time.time()))

    def set_param(self, x):
        if self.w is None:
            shape = x.shape
            w_rows = np.prod(shape[0:3])
            w_cols = self.b.shape[1]

            if self.wtype == 'random':
                self.w = np.random.randn(w_rows, w_cols)
            elif self.wtype == 'direct':
                self.w = np.zeros((w_rows, w_cols), dtype=None)
                rand_idx_gen = random_index_generator(w_rows)

                for col in range(w_cols):
                    self.w[next(rand_idx_gen)][col] = 1.
            elif self.wtype == 'diff':
                self.w = np.zeros((w_rows, w_cols), dtype=None)
                rand_idx_gen = random_index_generator(w_rows)

                for col in range(w_cols):
                    self.w[next(rand_idx_gen)][col] = 1.
                    self.w[next(rand_idx_gen)][col] = -1.
            else:
                raise Exception('wtype="{}" is not supported'.format(self.wtype))

    # [kernel_height, kernel_width, kernel_channel, kernel_number]
    def __call__(self, x):
        if self.w is None:
            shape = x.shape
            w_rows = np.prod(shape[0:3])
            w_cols = self.b.shape[1]

            if self.wtype == 'random':
                self.w = np.random.randn(w_rows, w_cols)
            elif self.wtype == 'direct':
                self.w = np.zeros((w_rows, w_cols), dtype=None)
                rand_idx_gen = random_index_generator(w_rows)

                for col in range(w_cols):
                    self.w[next(rand_idx_gen)][col] = 1.
            elif self.wtype == 'diff':
                self.w = np.zeros((w_rows, w_cols), dtype=None)
                rand_idx_gen = random_index_generator(w_rows)

                for col in range(w_cols):
                    self.w[next(rand_idx_gen)][col] = 1.
                    self.w[next(rand_idx_gen)][col] = -1.
            else:
                raise Exception('wtype="{}" is not supported'.format(self.wtype))
            # print('w = ', self.w)
            # self.w = tf.convert_to_tensor(self.w, dtype='float32')

        y = tf.reduce_mean(x, axis=3)
        z = tf.reshape(y, (1, -1))
        return tf.reduce_sum(self.scale * tf.losses.binary_crossentropy(self.b, tf.sigmoid(tf.matmul(z, tf.cast(self.w, tf.float32)))))

    def get_config(self):
        return {
            'scale': self.scale,
            'b': self.b,
            'target_block': self.target_block,
            'wtype': self.wtype,
            'w': self.w,
            'no_embed': self.no_embed
        }
 
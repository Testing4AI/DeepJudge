import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import time
from adversarial_attacks import FGSM, PGD, CW_L2


parser = argparse.ArgumentParser(description='DeepJudge black-box test case generation process')
parser.add_argument('--model', required=True, type=str, help='victim model path')
parser.add_argument('--seeds', required=True, type=str, help='selected seeds path')
parser.add_argument('--method', default='pgd', type=str, help='adversarial attacks. choice: fgsm/pgd/cw')
parser.add_argument('--ep', default=0.1, type=float, help='for fgsm/pgd attack (perturbation bound)')
parser.add_argument('--iters', default=10, type=int, help='for pgd attack')
parser.add_argument('--confidence', default=5, type=float, help='for cw attack')
parser.add_argument('--cmin', default=0, type=float, help='clip lower bound')
parser.add_argument('--cmax', default=1, type=float, help='clip upper bound')
parser.add_argument('--output', default='./testcases', type=str, help='test cases saved dir')


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


if __name__ == '__main__':
    opt = parser.parse_args()

    # load the victim model
    model_owner = load_model(opt.model)

    # load the seeds
    with np.load(opt.seeds) as f:
        seeds_x = f['seeds_x']
        seeds_y = f['seeds_y']
    
    start = time.time()
    if opt.method == 'fgsm':
        fgsm = FGSM(model_owner, ep=opt.ep)
        advx, advy = fgsm.generate(seeds_x, seeds_y)
    elif opt.method == 'pgd':
        pgd = PGD(model_owner, ep=opt.ep, epochs=opt.iters)
        advx, advy = pgd.generate(seeds_x, seeds_y)
    else:
        cw_params = {'batch_size': 50,
                     'confidence': opt.confidence,
                     'targeted': False,
                     'learning_rate': 0.001,
                     'binary_search_steps': 3,
                     'max_iterations': 1000,
                     'abort_early': True,
                     'initial_const': 0.01,
                     'clip_min': 0,
                     'clip_max': 1,
                     'shape': seeds_x.shape[1:]}
        cw = CW_L2(model_owner, **cw_params)
        advx, advy = cw.attack(seeds_x, seeds_y)
    print("TIME COST", time.time()-start)
    
    log_dir = opt.output
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    if opt.method == 'fgsm':
        save_path = f'{log_dir}/{opt.method}_ep{opt.ep}.npz'
    elif opt.method == 'pgd':
        save_path = f'{log_dir}/{opt.method}_ep{opt.ep}_iters{opt.iters}.npz'
    else:
        save_path = f'{log_dir}/{opt.method}_c{opt.confidence}.npz'
    
    np.savez(save_path, advx=advx, advy=advy)
    print('Black-box test cases saved at ' + save_path)



import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from metrics import Rob, JSD
        
parser = argparse.ArgumentParser(description='DeepJudge black-box metric evaluation')
parser.add_argument('--model', required=True, type=str, help='victim model path')
parser.add_argument('--suspect', required=True, type=str, help='suspect model path or a dir')
parser.add_argument('--tests', required=True, type=str, help='test case saved path')
parser.add_argument('--output', default='./results', type=str, help='evaluation results saved dir')


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
        log_file = log_dir + '/blackbox_evaluation.txt'
        with open(log_file, 'a') as f:
            print(content, file=f)   
        


if __name__ == '__main__':
    opt = parser.parse_args()
    
    # load the victim model 
    model_owner = load_model(opt.model)

    # load the black-box test cases
    with np.load(opt.tests) as f:
        advx = f['advx']
        advy = f['advy']
    
    log_dir = opt.output
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # metric evaluations
    if os.path.isfile(opt.suspect):
        model_suspect = load_model(opt.suspect)
        robd = Rob(model_suspect, advx, advy) 
        jsd = JSD(model_suspect, model_owner, advx)
        print(f"victim model:{opt.model}, suspect model: {opt.suspect}")
        print(f"RobD: {robd}, JSD: {jsd}")
        log(f"victim model:{opt.model}, suspect model: {opt.suspect}")
        log(f"RobD: {robd}, JSD: {jsd}")
    elif os.path.isdir(opt.suspect):
        for root, dirs, files in os.walk(opt.suspect):
            files.sort()
            for file in files:
                model_suspect = load_model(os.path.join(root, file))
                robd = Rob(model_suspect, advx, advy) 
                jsd = JSD(model_suspect, model_owner, advx)
                print(f"victim model:{opt.model}, suspect model: {file}")
                print(f"RobD: {robd}, JSD: {jsd}")
                log(f"victim model:{opt.model}, suspect model: {file}")
                log(f"RobD: {robd}, JSD: {jsd} \n")
                
     
    
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import random
import time

def create_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--guitar_dir', default = '../GuitarSet/audio_mono-mic-split/')
    parser.add_argument('--humming_dir', default = '../sample_hum/')
    parser.add_argument('--out_dir', default = "./output/")
    parser.add_argument('--name', default = "")
    parser.add_argument('--n_global_epochs', default = 100, type = int)
    parser.add_argument('--n_fixed_global_epochs', default = 10, type = int)
    parser.add_argument('--n_joint_epochs', default = 200, type = int)
    parser.add_argument('--plot_interval', default = 600, type=int)
    parser.add_argument('--log', default = 'log.txt')
    parser.add_argument('--seed', default = 999, type=int)
    parser.add_argument('--pretrained_global', default = None)
    parser.add_argument('--pretrained_local', default = None)
    
    config = parser.parse_args()
    return config


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 
def delete_all_files(fpath):
    if os.path.exists(fpath):
        for file in os.scandir(fpath):
            os.remove(file.path)
        return "remove all"
    else:
        return "directory not found"
    

def seed_worker(worker_id):
    worker_id = torch.initial_seed() % 2**32
    np.random.seed(worker_id)
    random.seed(worker_id)

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
        
def printlog(sent, path):
    f = open(path, "a")
    print(sent, file=f)
    time.sleep(.1)
    f.close()
    print(sent)
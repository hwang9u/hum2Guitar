import torch
import torch.nn as nn
import gc

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import soundfile as sf
import pickle
import os
from adabelief_pytorch import AdaBelief

from dataset import Hum2GuitarSet
from torch.utils.data import DataLoader
from model import GlobalGenerator, LocalEnhancer, MultiscaleDiscriminator, trainable_global, init_weights
from criterion import LossFM, LossGAN, lr_lambda
from utils.env import create_config, create_folder, seed_worker, set_seed, printlog
from utils.vis import plot_sample
from args import mel_kwargs, stft_kwargs
from trainer import Trainer

config = create_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(config.seed)

FM_LAMBDA=10
lr = 2e-4
beta1 = .5
beta2 = .999
lr_decay_epochs=3

# DATASET
guitarset = Hum2GuitarSet(config.guitar_dir, input_type='guitar', stft_kwargs=stft_kwargs, mel_kwargs=mel_kwargs)
hummingset = Hum2GuitarSet(config.humming_dir, input_type='humming', stft_kwargs=stft_kwargs, mel_kwargs=mel_kwargs)
h_s,x_s, db_mm=hummingset[0]
h_s = h_s.unsqueeze(0).to(device)
x_s = x_s.unsqueeze(0).to(device)

# DATALOADER
train_dloader = DataLoader(guitarset, batch_size=1,shuffle=True, num_workers=4 if torch.cuda.is_available() else 0, worker_init_fn=seed_worker)

# TRAINING
## Global Generator Training

global_trainer = Trainer(config, train_dloader=train_dloader, stft_kwargs=stft_kwargs, mel_kwargs=mel_kwargs, stage="global", device=device)
global_trainer.create_optimizer(lr = lr, beta1=beta1, beta2=beta2)
global_trainer.create_scheduler(lr_decay_epochs=lr_decay_epochs)

if config.pretrained_local is not None:
    pass
else:        
    if config.pretrained_global is not None:
        printlog(f'\n[STAGE 1] Load Pre-trained Global Generator/Discriminator from {global_trainer.config.pretrained_global}', global_trainer.LOG_PATH)
        global_trainer.load_checkpoint(config.pretrained_global)
    print("[Start] Training Global Generator\n" )
    global_trainer.train(h_s=h_s, x_s=x_s, FM_LAMBDA=FM_LAMBDA)
    print("[End] Training Global Generator\n" )

print("[Start] Training Local Enhancer\n" )
global_generator = global_trainer.net_G
local_trainer =  Trainer(config, train_dloader=train_dloader, global_generator=global_generator, stage="local", stft_kwargs=stft_kwargs, mel_kwargs=mel_kwargs, device = device)
local_trainer.create_optimizer(lr = lr, beta1=beta1, beta2=beta2)
local_trainer.create_scheduler(lr_decay_epochs=lr_decay_epochs)
if config.pretrained_local is not None:
    printlog(f'\n[STAGE 2] Load Pre-trained Local Enhancer/Discriminator from {config.pretrained_local}', local_trainer.LOG_PATH)
    local_trainer.load_checkpoint(config.pretrained_local)
local_trainer.train(h_s=h_s, x_s=x_s)


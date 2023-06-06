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


config = create_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(config.seed)

GUITAR_DIR = config.guitar_dir
HUMMING_DIR = config.humming_dir
OUT_DIR = config.out_dir
PLOT_INTERVAL = config.plot_interval
CHECKPOINT_SAVE_DIR = os.path.join(OUT_DIR, 'checkpoint/')
GUITAR_SAMPLE_SAVE_DIR = os.path.join(OUT_DIR, 'sample/', 'guitar/')
HUMMING_SAMPLE_SAVE_DIR = os.path.join(OUT_DIR, 'sample/', 'humming/')
LOG_PATH = os.path.join(config.out_dir, config.log)

N_GG_EPOCHS = config.n_global_epochs
N_FIXED_GLOBAL_EPOCHS = config.n_fixed_global_epochs
N_JOINT_EPOCHS = config.n_joint_epochs


create_folder(OUT_DIR)
create_folder(CHECKPOINT_SAVE_DIR)
create_folder(GUITAR_SAMPLE_SAVE_DIR)
create_folder(HUMMING_SAMPLE_SAVE_DIR)

printlog(f'device: {device}', LOG_PATH)
printlog(f'dataset: {GUITAR_DIR}', LOG_PATH)
printlog(f'output directory: {OUT_DIR}', LOG_PATH)
printlog(f'random seed: {config.seed}', LOG_PATH)
printlog(f'Training Epochs: Global Generator [{N_GG_EPOCHS}] Local Only [{N_FIXED_GLOBAL_EPOCHS}] Global + Local [{N_JOINT_EPOCHS}]', LOG_PATH)
print('\n')


# DATASET
guitarset = Hum2GuitarSet(GUITAR_DIR, input_type='guitar', stft_kwargs=stft_kwargs, mel_kwargs=mel_kwargs)
hummingset = Hum2GuitarSet(HUMMING_DIR, input_type='humming', stft_kwargs=stft_kwargs, mel_kwargs=mel_kwargs)
h_s,x_s, db_mm=hummingset[0]
h_s = h_s.unsqueeze(0).to(device)
x_s = x_s.unsqueeze(0).to(device)

# DATALOADER
train_dloader = DataLoader(guitarset, batch_size=1,shuffle=True, num_workers=4 if torch.cuda.is_available() else 0, worker_init_fn=seed_worker)
N =  len(train_dloader)
beta1 = 0.5
beta2 = .999
lr = 2*1e-4
eps = 1e-14
num_D=3
FM_LAMBDA = 10
DECAY_EPOCH = 100
var = .1

# LOSS FUNCTION
crit_gan = LossGAN(ls=True)
crit_fm = LossFM(num_D=num_D, num_layer=4)

########## [STAGE 1] TRAIN GLOBAL GENERATOR ##########
## MODEL
net_D = MultiscaleDiscriminator(input_dim=1,
                                    n_layers=4,
                                    use_sigmoid=False,
                                    return_inter_features=True,
                                    num_D=num_D).to(device)
net_GG = GlobalGenerator(gin_dim=1).to(device)

net_GG.apply(init_weights)
net_D.apply(init_weights)

## OPTIMIZER
optimizer_D = AdaBelief(net_D.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, print_change_log=False, weight_decouple=False, rectify=False )
optimizer_GG = AdaBelief(net_GG.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, print_change_log=False, weight_decouple=False, rectify=False)

## SCHEDULER
scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lambda epoch: lr_lambda(epoch, n_epochs=N_JOINT_EPOCHS+N_FIXED_GLOBAL_EPOCHS, decay_epoch=DECAY_EPOCH), verbose = True)
scheduler_GG = torch.optim.lr_scheduler.LambdaLR(optimizer_GG, lambda epoch: lr_lambda(epoch, n_epochs=N_JOINT_EPOCHS+N_FIXED_GLOBAL_EPOCHS, decay_epoch=DECAY_EPOCH), verbose = True)



loss_dict = {'G': [], 'D': []}
best_loss_G = np.inf
plot_ind = 0 
epoch_ = 0



if config.pretrained_global is not None:
    printlog(f'\n[STAGE 1] Load Pre-trained Global Generator/Discriminator from {config.pretrained_global}', LOG_PATH)
    checkpoint_global = torch.load(config.pretrained_global, map_location=device)
    epoch_ = checkpoint_global['epoch']
    print('load chekcpoint {} epoch'.format(epoch_))
    plot_ind = checkpoint_global['plot_ind']
    loss_dict = checkpoint_global['loss_dict_global']
    
    net_GG.load_state_dict(checkpoint_global['net_GG'])
    net_D.load_state_dict(checkpoint_global['net_D'])
    
    optimizer_D.load_state_dict(checkpoint_global['optimizer_D'])
    optimizer_GG.load_state_dict(checkpoint_global['optimizer_GG'])
    
    scheduler_GG.load_state_dict(checkpoint_global['scheduler_GG'])
    scheduler_D.load_state_dict(checkpoint_global['scheduler_D'])

    
for e in range(epoch_, N_GG_EPOCHS):
    if e == epoch_:
        printlog(f'\n[STAGE 1] Train Global Generator for {config.n_global_epochs} epochs\n', LOG_PATH)
    loss_D = 0
    loss_G = 0
    loss_G_GAN = 0
    loss_G_FM = 0
    loss_D_REAL=0
    loss_D_FAKE=0
    with tqdm(train_dloader, unit='batch') as tepoch:
        for b, (h, x, _) in enumerate(tepoch):       
            tepoch.set_description(f"[Epoch {e+1}]")

            net_GG.train()
            net_D.train()
            
            h = h.to(device)
            x = x.to(device)
        
            # Downsampling 
            DownSampling = nn.AvgPool2d(kernel_size=(3,1), stride=(2,1), padding=[1, 0], count_include_pad=False)
            x_real_down = DownSampling(x)
            h_down = DownSampling(h)

            # Update Generator
            
            # x' = G(h)
            x_fake_down = net_GG(h_down)
            
            # D(x')
            noise = torch.normal(0, var, size=x_fake_down.shape).to(device)
            y_fake_down = net_D(x_fake_down + noise, h_down)
            
            # D(x)
            noise = torch.normal(0, var, size=x_fake_down.shape).to(device)
            y_real_down = net_D(x_real_down + noise, h_down)
            
            loss_gan_fake_g = 0
            for i in range(num_D): # lossGAN
                loss_gan_fake_g += crit_gan(y_fake_down[i][-1], real_label=True)
            loss_fm = crit_fm(y_fake_down, y_real_down) # lossFM
            loss_g = loss_gan_fake_g + FM_LAMBDA * loss_fm
            
            net_GG.zero_grad()
            loss_g.backward(retain_graph=True) 
            optimizer_GG.step()
            
            # Update Discriminator
            # D(x')
            noise = torch.normal(0, var, size=x_fake_down.shape).to(device)
            y_fake_down = net_D(x_fake_down.detach()+noise, h_down)
        
            loss_gan_real_d = 0
            loss_gan_fake_d = 0
            for i in range(num_D): 
                loss_gan_fake_d += crit_gan(y_fake_down[i][-1], real_label=False)
                loss_gan_real_d += crit_gan(y_real_down[i][-1], real_label=True)
            loss_d = loss_gan_fake_d + loss_gan_real_d
            
            net_D.zero_grad()
            loss_d.backward()
            optimizer_D.step()
            
            loss_G += loss_g.item()
            loss_D += loss_d.item()

            loss_D_REAL += loss_gan_real_d.item()
            loss_D_FAKE += loss_gan_fake_d.item()
            
            loss_G_GAN += loss_gan_fake_g.item()
            loss_G_FM += loss_fm.item()
            
            tepoch.set_postfix(loss_d_fake=loss_gan_fake_d.item(), loss_d_real=loss_gan_real_d.item(), loss_g_fake=loss_gan_fake_g.item(), loss_fm=loss_fm.item())
            if b % PLOT_INTERVAL == 0:
                    net_GG.eval()
                    with torch.no_grad():
                        fig = plot_sample(h=h_down, x=x_real_down, model=net_GG, input_type='guitar', stft_kwargs=stft_kwargs, mel_kwargs=mel_kwargs)
                        fig.suptitle(f'Epoch {e+1}')
                        fig.tight_layout()
                        fig.savefig(f'{GUITAR_SAMPLE_SAVE_DIR}/gg_{str(plot_ind).zfill(4)}.jpg')
                        plt.close(fig)
                        plot_ind += 1

    scheduler_D.step()
    scheduler_GG.step()
    
    epoch_result = f"[Epoch {e+1}] loss D: {loss_D /N} [REAL: {loss_D_REAL/N} FAKE: {loss_D_FAKE/N}] loss G: {loss_G /N} [GAN: {loss_G_GAN/N } + FM: {loss_G_FM/N }]"  
    printlog(epoch_result, LOG_PATH)    
    
    # Save loss dict
    loss_dict['D'].append(loss_D / len(train_dloader))    
    loss_dict['G'].append(loss_G / len(train_dloader))    
    
    
    # Save checkpoint
    checkpoint_global = {
            'epoch': e+1,
            'plot_ind': plot_ind,
            'net_GG': net_GG.state_dict(),
            'net_D': net_D.state_dict(),
            'optimizer_GG': optimizer_GG.state_dict(),
            'scheduler_GG': scheduler_GG.state_dict(),
            'scheduler_D': scheduler_D.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'loss_dict_global': loss_dict,
            'loss_G': loss_G,
            }
    torch.save(checkpoint_global, f"{CHECKPOINT_SAVE_DIR}/checkpoint_global{config.name}.tar") ## save checkpoint
    best_loss_G = loss_G
    
    gc.collect()
    torch.cuda.empty_cache() 
    

########## [STAGE 2] TRAIN LOCAL ENHANCER ##########
printlog(f'\n[STAGE 2] Train Local Enhancer for {N_FIXED_GLOBAL_EPOCHS + N_JOINT_EPOCHS} epochs', LOG_PATH)
## MODEL
net_D = MultiscaleDiscriminator(input_dim=1,
                                    n_layers=4,
                                    use_sigmoid=False,
                                    return_inter_features=True,
                                    num_D=num_D).to(device)
net_LE = LocalEnhancer(input_dim=1,
                       global_generator=None).to(device)
net_D.apply(init_weights)
net_LE.apply(init_weights)
net_LE.global_generator = nn.Sequential(*[getattr(net_GG, k) for k in ['down', 'res', 'up']])

## OPITMIZER
optimizer_D = AdaBelief(net_D.parameters(), lr=lr, betas=(beta1, beta2), eps=eps,print_change_log=False, weight_decouple=False, rectify=False)
optimizer_LE = AdaBelief(net_LE.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, print_change_log=False, weight_decouple=False, rectify=False)

## SCHEDULER
scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lambda epoch: lr_lambda(epoch, n_epochs=N_JOINT_EPOCHS+N_FIXED_GLOBAL_EPOCHS, decay_epoch=DECAY_EPOCH), verbose = True)
scheduler_LE = torch.optim.lr_scheduler.LambdaLR(optimizer_LE, lambda epoch: lr_lambda(epoch, n_epochs=N_JOINT_EPOCHS+N_FIXED_GLOBAL_EPOCHS, decay_epoch=DECAY_EPOCH), verbose = True)

loss_dict = {'G': [], 'D': []}
best_loss_G = np.inf
plot_ind = 0 
epoch_ = 0

if config.pretrained_local is not None:
    printlog(f'\n[STAGE 2] Load Pre-trained Local Enhancer/Discriminator from {config.pretrained_local}', LOG_PATH)
    checkpoint_local = torch.load(config.pretrained_local, map_location=device)
    epoch_ = checkpoint_local['epoch']
    print('load chekcpoint {} epoch'.format(epoch_))
    plot_ind = checkpoint_local['plot_ind']
    loss_dict = checkpoint_local['loss_dict_local']
    
    net_LE.load_state_dict(checkpoint_local['net_LE'])
    net_D.load_state_dict(checkpoint_local['net_D'])
    
    optimizer_D.load_state_dict(checkpoint_local['optimizer_D'])
    optimizer_LE.load_state_dict(checkpoint_local['optimizer_LE'])
    
    scheduler_LE.load_state_dict(checkpoint_local['scheduler_LE'])
    scheduler_D.load_state_dict(checkpoint_local['scheduler_D'])
    


for e in range(epoch_, N_FIXED_GLOBAL_EPOCHS + N_JOINT_EPOCHS):
    loss_D = 0
    loss_G = 0
    
    loss_G_GAN = 0
    loss_G_FM = 0
    
    loss_D_REAL=0
    loss_D_FAKE=0

    if (N_FIXED_GLOBAL_EPOCHS > 0) & (e == epoch_):
        printlog(f"--> Train Only Local Enhancer for {N_FIXED_GLOBAL_EPOCHS} epochs", LOG_PATH)
        trainable_global(net_LE, grad=False) 
    elif e == (epoch_ + N_FIXED_GLOBAL_EPOCHS):
        printlog(f"--> Train Global Generator and Local Enhancer for {N_JOINT_EPOCHS} epochs",LOG_PATH)
        trainable_global(net_LE, grad=True)
    else:
        pass
    
    with tqdm(train_dloader, unit='batch') as tepoch:
        for b, (h, x, db_mm) in enumerate(tepoch):
            tepoch.set_description(f"[Epoch {e+1}]")
            
            net_LE.train()
            net_D.train()
            
            h = h.to(device)
            x = x.to(device)
            
            # x' = G(h)        
            x_fake = net_LE(h)
            
            # D(x')
            noise = torch.normal(0, var, size=x_fake.shape).to(device)
            y_fake = net_D(x_fake + noise, h)
            
            # D(x)
            noise = torch.normal(0, var, size=x_fake.shape).to(device)
            y_real = net_D(x + noise, h)

            # Update Generator
            loss_gan_fake_g = 0
            for i in range(num_D): 
                loss_gan_fake_g += crit_gan(y_fake[i][-1], real_label=True)
            loss_fm = crit_fm(y_fake, y_real) 
            loss_g = loss_gan_fake_g + FM_LAMBDA * loss_fm
        
            net_LE.zero_grad()
            loss_g.backward(retain_graph=True)
            optimizer_LE.step()

            # Update Discriminator
            # D(x')
            noise = torch.normal(0, var, size=x_fake.shape).to(device)
            y_fake = net_D(x_fake.detach()+noise, h)
            
            loss_gan_real_d = 0
            loss_gan_fake_d = 0
            for i in range(num_D): 
                loss_gan_fake_d += crit_gan(y_fake[i][-1], real_label=False)
                loss_gan_real_d += crit_gan(y_real[i][-1], real_label=True)
            loss_d = loss_gan_fake_d + loss_gan_real_d
            
            net_D.zero_grad()
            loss_d.backward()
            optimizer_D.step()
        
        
            loss_G += loss_g.item()
            loss_D += loss_d.item()
            
            loss_D_REAL += loss_gan_real_d.item()
            loss_D_FAKE += loss_gan_fake_d.item()
            
            loss_G_GAN += loss_gan_fake_g.item()
            loss_G_FM += loss_fm.item()
            
            tepoch.set_postfix(loss_d_fake=loss_gan_fake_d.item(), loss_d_real=loss_gan_real_d.item(), loss_g_fake=loss_gan_fake_g.item(), loss_fm=loss_fm.item())
            
            if b % PLOT_INTERVAL == 0:
                    net_LE.eval()
                    with torch.no_grad():
                        fig = plot_sample(h=h, x=x
                                        , model=net_LE, input_type='guitar', stft_kwargs=stft_kwargs, mel_kwargs=mel_kwargs)
                        fig.suptitle(f'Epoch {e+1}')
                        fig.tight_layout()
                        fig.savefig(f'{GUITAR_SAMPLE_SAVE_DIR}/le_sample_{str(plot_ind).zfill(4)}.jpg')
                        plt.close(fig)
                        
                        fig = plot_sample(h=h_s, x=x_s
                                        , model=net_LE, input_type='humming', stft_kwargs=stft_kwargs, mel_kwargs=mel_kwargs)
                        fig.suptitle(f'Epoch {e+1}')
                        fig.tight_layout()
                        fig.savefig(f'{HUMMING_SAMPLE_SAVE_DIR}/le_sample_{str(plot_ind).zfill(4)}.jpg')
                        plt.close(fig)
                        plot_ind += 1        
    scheduler_D.step()
    scheduler_LE.step()
    
    
    loss_dict['D'].append(loss_D / len(train_dloader))    
    loss_dict['G'].append(loss_G / len(train_dloader))    
   
    
    epoch_result = f"[Epoch {e+1}] loss D: {loss_D /N} [REAL: {loss_D_REAL/N} FAKE: {loss_D_FAKE/N}] loss G: {loss_G /N} [GAN: {loss_G_GAN/N } + FM: {loss_G_FM/N }]"  
    printlog(epoch_result, LOG_PATH)    

    checkpoint_local = {
            'epoch': e+1,
            'plot_ind': plot_ind,
            'net_LE': net_LE.state_dict(),
            'net_D': net_D.state_dict(),
            'optimizer_LE': optimizer_LE.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'scheduler_LE': scheduler_LE.state_dict(),
            'scheduler_D': scheduler_D.state_dict(),
            'loss_dict_local': loss_dict,
            'loss_G': loss_G,
            }
    torch.save(checkpoint_local, f"{CHECKPOINT_SAVE_DIR}/checkpoint_local{config.name}.tar")
    best_loss_G = loss_G

    gc.collect()
    torch.cuda.empty_cache()    
    

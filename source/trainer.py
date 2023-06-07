import torch
import torch.nn as nn
import gc
from utils.env import printlog
import os
from tqdm import tqdm
from adabelief_pytorch import AdaBelief
from model import GlobalGenerator, LocalEnhancer, MultiscaleDiscriminator, init_weights, trainable_global
from criterion import LossFM, LossGAN, lr_lambda
from utils.vis import plot_sample
from utils.env import create_folder
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, config, train_dloader, stft_kwargs, mel_kwargs, global_generator=None, stage="global", device= "cpu", lr_decay_epochs=100):
        self.device = device
        self.stage=stage
        self.config = config
        self.stft_kwargs = stft_kwargs
        self.mel_kwargs = mel_kwargs
        self.train_dloader = train_dloader
        self.N = len(train_dloader)
        self.n_epochs = config.n_global_epochs if self.stage=="global" else (config.n_fixed_global_epochs + config.n_joint_epochs)
        self.create_ev()

        self.plot_ind = 0
        self.epoch_ = 0
        self.loss_dict = {'G': [], 'D': []}
        
        self.optimizer_G = None
        self.optimizer_D = None

        self.scheduler_G = None
        self.scheduler_D = None

        
        ## Discriminator
        self.net_D = MultiscaleDiscriminator(input_dim=1, n_layers=4, use_sigmoid=False, num_D=3, return_inter_features=True)
        self.net_D.apply(init_weights)        
        
        ## Generator
        self.net_G = GlobalGenerator(gin_dim=1)
        self.net_G.apply(init_weights)
        
        if stage == "local":
            net_LE = LocalEnhancer(input_dim=1, global_generator=None)
            net_LE.apply(init_weights)
            net_LE.global_generator = nn.Sequential(*[getattr(global_generator, k) for k in ['down', 'res', 'up']])
            self.net_G = net_LE

        
        
        self.net_G.to(device)
        self.net_D.to(device)
                        
        self.crit_gan = LossGAN(ls=True)
        self.crit_fm = LossFM(num_D=self.net_D.num_D, n_layers=self.net_D.n_layers)
        self.DownSampling = nn.AvgPool2d(kernel_size=(3,1), stride=(2,1), padding=[1, 0], count_include_pad=False)
            
    def create_optimizer(self, lr, beta1=0.5, beta2=0.999, eps=1e-12):
        self.optimizer_D = AdaBelief(self.net_D.parameters(), lr=lr, betas=(beta1, beta2), eps=eps,print_change_log=False, weight_decouple=False, rectify=False)
        self.optimizer_G = AdaBelief(self.net_G.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, print_change_log=False, weight_decouple=False, rectify=False)

    def create_scheduler(self, lr_decay_epochs = 100):
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lambda epoch: lr_lambda(epoch, n_epochs=self.n_epochs, decay_epoch=lr_decay_epochs), verbose = True)
        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lambda epoch: lr_lambda(epoch, n_epochs=self.n_epochs, decay_epoch=lr_decay_epochs), verbose = True)

    def create_ev(self):
        self.LOG_PATH = os.path.join(self.config.out_dir, self.config.log)
        self.GUITAR_SAMPLE_SAVE_DIR = os.path.join(self.config.out_dir, 'sample/', 'guitar/')
        self.HUMMING_SAMPLE_SAVE_DIR = os.path.join(self.config.out_dir, 'sample/', 'humming/')
        self.CHECKPOINT_SAVE_DIR = os.path.join(self.config.out_dir, 'checkpoint/')
        create_folder(self.config.out_dir)
        create_folder(self.CHECKPOINT_SAVE_DIR)
        create_folder(self.GUITAR_SAMPLE_SAVE_DIR)
        create_folder(self.HUMMING_SAMPLE_SAVE_DIR)

        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device) 
        self.epoch_ = checkpoint['epoch']
        print('load chekcpoint {} epoch'.format(self.epoch_))
        
        self.plot_ind = checkpoint['plot_ind']
        self.loss_dict = checkpoint['loss_dict']
        
        self.net_G.load_state_dict(checkpoint['net_G'])
        self.net_D.load_state_dict(checkpoint['net_D'])
        
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D'])


    def train(self, h_s, x_s, var= 0., FM_LAMBDA=10):        
        for e in range(self.epoch_, self.n_epochs):
            if self.stage =="local":
                if (self.config.n_fixed_global_epochs> 0) & (e == self.epoch_):
                    printlog(f"--> Train Only Local Enhancer for {self.config.n_fixed_global_epochs} epochs", self.LOG_PATH)
                    trainable_global(self.net_G, grad=False) 
                if ((e == (self.epoch_ + self.config.n_fixed_global_epochs)) | (self.epoch_ > self.config.n_fixed_global_epochs)) & (e == self.epoch_):
                    printlog(f"--> Train Global Generator and Local Enhancer for {self.config.n_joint_epochs} epochs", self.LOG_PATH)
                    trainable_global(self.net_G, grad=True)
                else:
                    pass
            
            loss_D = 0
            loss_G = 0
            loss_G_GAN = 0
            loss_G_FM = 0
            loss_D_REAL=0
            loss_D_FAKE=0



            with tqdm(self.train_dloader, unit='batch') as tepoch:
                for b, (h, x, _) in enumerate(tepoch):       
                    tepoch.set_description(f"[Epoch {e+1}]")

                    self.net_G.train()
                    self.net_D.train()

                    h = h.to(self.device)
                    x = x.to(self.device)

                    if self.stage =="global":
                        # Downsampling 
                        x = self.DownSampling(x).detach()
                        h = self.DownSampling(h).detach()
        
                    # Update Generator

                    # x' = G(h)
                    x_fake = self.net_G(h)

                    # D(x')
                    noise = torch.normal(0, var, size=x_fake.shape).to(self.device)
                    y_fake = self.net_D(x_fake + noise, h)

                    # D(x)
                    y_real = self.net_D(x + noise, h)

                    loss_gan_fake_g = 0
                    for i in range(self.net_D.num_D): # lossGAN
                        loss_gan_fake_g += self.crit_gan(y_fake[i][-1], real_label=True)
                    loss_fm = self.crit_fm(y_fake, y_real) # lossFM
                    loss_g = loss_gan_fake_g + FM_LAMBDA * loss_fm

                    self.net_G.zero_grad()
                    loss_g.backward(retain_graph=True) 
                    self.optimizer_G.step()

                    # Update Discriminator
                    # D(x')
                    noise = torch.normal(0, var, size=x_fake.shape).to(self.device)
                    y_fake = self.net_D(x_fake.detach()+noise, h)

                    loss_gan_real_d = 0
                    loss_gan_fake_d = 0
                    for i in range(self.net_D.num_D): 
                        loss_gan_fake_d += self.crit_gan(y_fake[i][-1], real_label=False)
                        loss_gan_real_d += self.crit_gan(y_real[i][-1], real_label=True)
                    loss_d = loss_gan_fake_d + loss_gan_real_d

                    self.net_D.zero_grad()
                    loss_d.backward()
                    self.optimizer_D.step()

                    loss_G += loss_g.item()
                    loss_D += loss_d.item()

                    loss_D_REAL += loss_gan_real_d.item()
                    loss_D_FAKE += loss_gan_fake_d.item()

                    loss_G_GAN += loss_gan_fake_g.item()
                    loss_G_FM += loss_fm.item()

                    tepoch.set_postfix(loss_d_fake=loss_gan_fake_d.item(), loss_d_real=loss_gan_real_d.item(), loss_g_fake=loss_gan_fake_g.item(), loss_fm=loss_fm.item())
                    if b % self.config.plot_interval == 0:
                            self.net_G.eval()

                            with torch.no_grad():
                                fig = plot_sample(h=h, x=x
                                                , model=self.net_G, input_type='guitar', stft_kwargs=self.stft_kwargs, mel_kwargs=self.mel_kwargs)
                                fig.suptitle(f'Epoch {e+1}')
                                fig.tight_layout()
                                fig.savefig(f'{self.GUITAR_SAMPLE_SAVE_DIR}/{self.stage}_sample_{str(self.plot_ind).zfill(4)}.jpg')
                                plt.close(fig)

                                fig = plot_sample(h=h_s, x=x_s
                                                , model=self.net_G, input_type='humming', stft_kwargs=self.stft_kwargs, mel_kwargs=self.mel_kwargs)
                                fig.suptitle(f'Epoch {e+1}')
                                fig.tight_layout()
                                fig.savefig(f'{self.HUMMING_SAMPLE_SAVE_DIR}/{self.stage}_sample_{str(self.plot_ind).zfill(4)}.jpg')
                                plt.close(fig)
                                self.plot_ind += 1   
                                del fig
                    del x
                    del h
                    del y_fake
                    del y_real
                    del loss_gan_fake_d
                    del loss_gan_fake_g
                    del loss_gan_real_d
                    del x_fake
                    del loss_d
                    del loss_g

            if self.scheduler_D != None:
                self.scheduler_D.step()
            if self.scheduler_G != None:
                self.scheduler_G.step()

            epoch_result = f"[Epoch {e+1}] loss D: {loss_D /self.N} [REAL: {loss_D_REAL/self.N} FAKE: {loss_D_FAKE/self.N}] loss G: {loss_G /self.N} [GAN: {loss_G_GAN/self.N } + FM: {loss_G_FM/self.N }]"  
            printlog(epoch_result, self.LOG_PATH)    

            # Save loss dict
            self.loss_dict['D'].append(loss_D / self.N)
            self.loss_dict['G'].append(loss_G / self.N)


            # Save checkpoint
            checkpoint = {
                    'epoch': e+1,
                    'plot_ind': self.plot_ind,
                    'net_G': self.net_G.state_dict(),
                    'net_D': self.net_D.state_dict(),
                    'optimizer_G': self.optimizer_G.state_dict(),
                    'optimizer_D': self.optimizer_D.state_dict(),
                    'scheduler_G': self.scheduler_G.state_dict(),
                    'scheduler_D': self.scheduler_D.state_dict(),
                    'loss_dict': self.loss_dict,
                    }
            torch.save(checkpoint, f"{self.CHECKPOINT_SAVE_DIR}/checkpoint_{self.stage}{self.config.name}.tar") ## save checkpoint
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache() 









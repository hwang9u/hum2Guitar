import librosa
import matplotlib.pyplot as plt
import torch
import numpy as np
import re

def plot_sample(h, x, input_type,  stft_kwargs, mel_kwargs, model=None, y=None): 
    '''
    args:
        h: semantic mel spectrogram
        x: input mel spectrogram
    '''
    if y is None:
        model.eval()
        with torch.no_grad():
            y = model(h).detach().cpu().squeeze().numpy()
            
    if not isinstance(h, np.ndarray):
        h = h.detach().cpu().squeeze().numpy()
    if not isinstance(x, np.ndarray):
        x = x.detach().cpu().squeeze().numpy()
        
    
    fig, axes = plt.subplots(1,3, figsize = (18,5))
    im0 = librosa.display.specshow(h, hop_length=stft_kwargs['hop_length'], x_axis='time', y_axis='mel', cmap='magma', ax=axes[0], fmax = mel_kwargs['fmax'], fmin= mel_kwargs['fmin'])
    im1 = librosa.display.specshow(x, hop_length=stft_kwargs['hop_length'], x_axis='time', y_axis='mel', cmap='magma', ax=axes[1], fmax = mel_kwargs['fmax'], fmin= mel_kwargs['fmin'])
    im2 = librosa.display.specshow( y,
                                   hop_length=stft_kwargs['hop_length'], x_axis='time', y_axis='mel', cmap = 'magma', ax=axes[2], fmax = mel_kwargs['fmax'], fmin= mel_kwargs['fmin'])
    
    fig.colorbar(im0, ax=axes[0])
    fig.colorbar(im1, ax=axes[1])
    fig.colorbar(im2, ax=axes[2])
    
    axes[0].set_title('Input')
    axes[1].set_title('Original Humming' if input_type == 'humming' else 'Ground truth')
    axes[2].set_title('Output')
    
    fig.tight_layout()
    return fig

def plot_loss_curve(loss_dict, suptitle  = 'Loss Curve'):
    fig, axes = plt.subplots(1,2, figsize = (15, 3), dpi=100)
    axes[0].plot(np.arange(len(loss_dict['G']))+1, loss_dict['G'], c='blue')
    axes[0].set_xlabel('Epochs')
    axes[0].set_title('loss G')
    axes[0].grid()


    axes[1].plot(np.arange(len(loss_dict['D']))+1, loss_dict['D'], c='red')
    axes[1].set_xlabel('Epochs')
    axes[1].set_title('loss D')
    axes[1].grid()
    fig.suptitle(suptitle)
    plt.tight_layout()
    return fig




def get_loss_log(log_path):
    with open(log_path, 'r') as f:
        log_txt = f.readlines()
    
    log = list(filter(lambda x: "Epoch" in x, log_txt))
    log_list = list(map(lambda l: re.match('\\[Epoch [0-9]*\\] .+ \\[REAL: (.+) FAKE: (.+)] loss G: .+ \\[GAN: (.+) . FM: (.+)]',
                           string=l).groups(), log))
    
    log_arr = np.array(log_list).astype('float')
    return log_arr
    
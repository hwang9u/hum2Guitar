from utils.normalize import denormalize
import torch
import librosa
from args import stft_kwargs, mel_kwargs

def hum_to_guitar(h_s, model, db_mm,stft_kwargs, mel_kwargs, return_mel=False):
    model.eval()
    with torch.no_grad():
        h_ = model(h_s).detach().cpu().squeeze().numpy()
    h_ = denormalize(h_, db_mm)
    h_ = librosa.db_to_power(h_)
    sig_h = librosa.feature.inverse.mel_to_audio(h_,
                                                 center=True,
                                                 n_fft=stft_kwargs['n_fft'],hop_length=stft_kwargs['hop_length'], win_length=stft_kwargs['win_length'],
                                                 fmax=mel_kwargs['fmax'], fmin=mel_kwargs['fmin']
                                                 )
    if return_mel:
        return sig_h, h_
    else:
        return sig_h
    
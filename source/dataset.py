## Dataset
import librosa
import numpy as np
import numpy as np
import torch
import os
import glob
from torch.utils.data import Dataset
from utils.harmonic import cepsf0, pitchsyn
from utils.normalize import normalize, minmax


class Hum2GuitarSet(Dataset):
    def __init__(self, dataset_dir, flist = None, input_type='humming',
                 stft_kwargs={'n_fft':2048, 'win_length':1024, 'hop_length': 512},
                 mel_kwargs={'n_mels': 256, 'fmax':8000, 'fmin': 20},
                 sr=22050 ):
        self.input_type=input_type
        self.dataset_dir = dataset_dir
        self.flist = glob.glob( os.path.join(dataset_dir, '*solo*.wav' if type=='guitar' else '*.wav') ) if flist is None else flist
        self.sr = sr
        self.stft_kwargs = stft_kwargs
        self.mel_kwargs = mel_kwargs
            
    def __getitem__(self, index):
        # Loaad audio signal
        sig, sr = librosa.load(self.flist[index], sr=self.sr)

        h, x, db_mm = preprocessing(sig, sr=sr, stft_kwargs=self.stft_kwargs, mel_kwargs=self.mel_kwargs, input_type=self.input_type)
        return h, x, db_mm

    def __len__(self):
        return len(self.flist)
    
    
def preprocessing(sig=None, fn=None, stft_kwargs={}, mel_kwargs={}, sr=22050, input_type='guitar'):
    if fn is not None:
        sig, sr = librosa.load(fn, sr=sr)
    ## STFT
    S = librosa.stft(y=sig, **stft_kwargs)
    M = np.abs(S) # magnitude
    D = librosa.amplitude_to_db(M) # magnitude(dB)
    P = M**2 # power 
    
    if input_type == 'humming':
        pitch, pitch_ind = cepsf0(D=D, sr=sr, fmax =  550., fmin=70, verbose=False, remove_outliers=True)
    elif input_type == 'guitar':
        pitch, pitch_ind = cepsf0(D=D, sr=sr, fmax = 1300., fmin=20, verbose=False)        
    else:
        raise ValueError("input_type must be one of {'humming', 'guitar'}")

    Mel = librosa.power_to_db(librosa.feature.melspectrogram(S=P, **mel_kwargs))
    Mel, db_mm = normalize(Mel, return_minmax_values=True)

    # Generating semantic harmonics using pitch(decreasing magnitude as the frequency increases)
    l=1.0 ; decay = l * np.exp(-l * np.linspace(0, 5, stft_kwargs['n_fft']//2+1 ))
    H = pitchsyn(M, pitch_ind=pitch_ind) * minmax(decay).reshape(-1,1)

    # STFT to Mel
    Mel_H = librosa.power_to_db(librosa.feature.melspectrogram(S=H**2, **mel_kwargs)) 
    Mel_H = normalize(Mel_H, return_minmax_values=False)
    return torch.Tensor(Mel_H).unsqueeze(0), torch.Tensor(Mel).unsqueeze(0), db_mm

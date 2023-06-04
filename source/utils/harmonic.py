import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import median_filter
import librosa
import librosa.display



## F0 estimation
## from pyceps: htttps://github.com/hwang9u/pyceps
def rceps(D, qmin_ind=0, qmax_ind = None, sr=22050):
    '''
    real cepstrum obtained from log spectrogram (dB),
    '''
    n_fft = int((D.shape[0]-1)*2)
    quef = np.arange(n_fft//2 + 1)/sr 
    if qmax_ind == None:
        qmax_ind = n_fft//2 +1
    C = np.apply_along_axis(func1d=lambda x: np.fft.irfft(x).real, axis=0, arr=D)
    C = C[qmin_ind: qmax_ind, :]
    quef = quef[qmin_ind: qmax_ind]
    return quef, C   

def find_closest_ind(x , ref):
    fftmat = np.tile(ref.reshape(-1,1), (1, x.size))
    x_mat = np.tile(  x.reshape(1,-1), (len(ref),1))
    x_ind = np.abs(fftmat - x_mat).argmin(axis=0)
    return x_ind

def find_max_db_ind(x,Dt,quef, lfrq, n=3):
    max_cand = np.argsort(x)[::-1][:n]
    freq_cand = 1/quef[max_cand]
    f0_ind_cand = find_closest_ind(freq_cand, lfrq)
    mag_cand = Dt[f0_ind_cand]
    return max_cand[mag_cand.argmax()]

def cepsf0(D, sr=22050, fmax = 400, fmin=0, win_size= 3, verbose=True, remove_outliers = False):
    '''
    f0 estimation from high quefrency of cepstrum
    '''
    n_fft = int((D.shape[0]-1)*2)
    lfrq = librosa.fft_frequencies(n_fft=n_fft, sr=sr) # linear frequency
    quef = (np.arange(n_fft//2 + 2)/sr)[1:] # quefrency (f: sr/2 ~ 0) w/o inf
    qmin_ind = np.where( (1/quef) <= fmax)[0][0]
    qmax_ind = np.where( (1/quef) >= fmin)[0][-1] if fmin > 1 else D.shape[0]-1
    if verbose:
        print("Search range: F0 [Hz] [{:.2f}, {:.2f}] / quefrency(index): [{:.5f}({}), {:.5f}({})] ".format(1/quef[qmin_ind], 1/quef[qmax_ind],
                                                                                                           quef[qmin_ind], qmin_ind,
                                                                                                           quef[qmax_ind], qmax_ind
                                                                                                           ))
        
    _, C = rceps(D, sr=sr, qmin_ind=qmin_ind, qmax_ind=qmax_ind) # real cepstrum
    qf0_ind = qmin_ind +  np.array(list(map(lambda t: find_max_db_ind(C[:,t], Dt=D[:,t], quef=_, lfrq=lfrq) , range(C.shape[1]) ) ))
    f0 = 1/quef[qf0_ind] # quefrency -> frequency
    
    # remove outliers using IQR method
    if remove_outliers: 
        med_f0 = np.median(f0)
        iqr = np.diff(np.quantile(f0, (.25,.75)))
        outlier_ind = np.where( np.abs(f0 - med_f0) <= 1.5*iqr, 1, 0 )
        f0 *= outlier_ind
    
    # median filtering
    if win_size > 0:
        f0 = median_filter(f0, win_size) # smoothing

    # mapping to linear frequency index
    f0_ind = find_closest_ind(f0, lfrq)    
    return f0, f0_ind


## Generate Harmonics on Frequency Domain using Pitch(f0)
def gaussian(x):
    return np.exp(-.5 * x**2)


def pitchsyn(M, pitch = None, pitch_ind = None, sr = 22050):
    D = librosa.amplitude_to_db(M)
    _n_fft, times=D.shape
    n_fft = int((_n_fft-1)*2) 
    lfrq = librosa.fft_frequencies(n_fft=n_fft, sr=sr)
    if pitch_ind is None:
        pitch_ind = find_closest_ind(pitch, lfrq)
    SynM = np.zeros_like(D) 
    for t in range(times):
        if (pitch_ind[t] <1):
            pass
        else:
            p = pitch_ind[t]
            harmonic_ind = [p*i  for i in range(1, _n_fft//p)]
            mm = (M[:,t]**2).sum()
            SynM[:, t] += gaussian(  4*(np.tile(np.arange(_n_fft).reshape(-1,1), (1, len(harmonic_ind))) - harmonic_ind )/2 ).sum(axis = 1) * mm
    return SynM
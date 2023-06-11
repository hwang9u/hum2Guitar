# Hum2Guitar
## Acoustic Guitar Timbre Transfer using Pix2PixHD architecture

<center>
<img src="./output/img/humming_sample_img_ex.jpg" width="1000px" height="250px">
</center>



## ✨ Inspiration
I have ALWAYS loved playing the Guitar🎸 since I was young. (I especially enjoy fingerstyle playing.) When I first started practicing the guitar, my fingers didn't work. So I once dreamed that *"If I hum, I want it to change to guitar sound."* It was wildest dream at the time, so I just practiced the guitar harder. 😂 This project began simply with curiosity about the memory.

## Approach
#### ✅ Timbre-Transfer using Image-to-Image translation techniques
* I utilized the mel spectrogram as the imput images. It can show the time-frequency characteristics of sound.
* However, since it has only magnitude information. So I used Griffin-Lim algorithm as a baseline for phase reconstruction.
#### ✅ Semantic Harmonics as input semantic label
* I attempted to extract the fundamental frequency(F0) from audio signal and utilize the fact that *"its positive integer multiple is a harmonic"* to create a semantic label.
* I'll refer to the artificially generated harmonics as "Semantic Harmonics"
* **Therefore, we can create our own paired dataset. As you may have guessed, not only humming but any sound with pitch can be transformed into a guitar sound!**

#### ✅ Pix2PixHD Architecture
  * At first, I attempted to use the Pix2Pix architecture, but I found that it didn't represent local information well, resulting in lack of sharpness in the output audio sound.
  * Therefore, I tried to employ the Pix2PixHD architecture, which is known for capturing fine-grained details in local information. (Other SOTA architectures are also worth trying.)

#### ✅ Data Segmentation Instead of Data Augmentation
  * There are only 180 "solo" samples available in the GuitarSet dataset. It is extremely small size. I though more audio samples with various pitch are required.
  * Although the dataset size was very small, I didn't do any augmentation. **Because playing low notes on the guitar is not simply a matter of lowering the pitch. The resonance when plucking the strings also varies. When playing low notes, there is more "buzzing" sound, but "pitch shift" did not reflect this aspect.**
  * Instead, I segmented the audio files into 5s durations and stored them to enable more weight updates.

<br>


### Data Pre-Processing

<center>
<img src="./output/img/diagram/preprocessing.png" width="1000px" height="200px">
</center>

### Training
<center>
<img src="./output/img/diagram/training.png" width="1000px" height="300px">
</center>

### Reconstructing the Audio signal from Synthesized Mel Spectrogram
<center>
<img src="./output/img/diagram/recon_to_audio.png" width="1000px" height="200px">
</center>


<br>

## Dataset
#### GuitarSet
  * only guitar "solo" (180 files)
  * link: https://github.com/marl/GuitarSet

#### MTG-QBH
  * This is used purely as evaluation examples and were never involved in the training process.
  * link: https://www.upf.edu/web/mtg/mtg-qbh

<br>

## Run
```python
$ cd hum2guitar
$ python source/train.py --guitar_dir GUITARSET_DIR --humming_dir HUMMING_DIR  
```
* Check ```utils/env.py``` and ```args.py``` for more training details.

<br>

## Results
#### ✔️ **[Check notebook and Listen examples here](https://nbviewer.org/gist/hwang9u/abf4fb685f10b435d88ba1f9f2eda822)**

### Humming to Guitar
<center>
<img src="./output/img/humming_sample_q116.jpg" width="1000px" height="150px">
<img src="./output/img/humming_sample_q9.jpg" width="1000px" height="150px">
<img src="./output/img/humming_sample_q66.jpg" width="1000px" height="150px">
<img src="./output/img/humming_sample_q107.jpg" width="1000px" height="150px">
</center>

### +) Guitar to Guitar
 #### How close is the sound restored from the semantic harmonics of the guitar to the original, not humming semantic harmonics?

<center>
<img src="./output/img/guitar_sample_00_Rock2-142-D_solo_mic.jpg" width="1000px" height="150px">
<img src="./output/img/guitar_sample_05_Funk3-98-A_solo_mic.jpg" width="1000px" height="150px">
<img src="./output/img/guitar_sample_05_SS3-98-C_solo_mic.jpg" width="1000px" height="150px">
</center>

* ❗️ We can guess on how well the model can accurately restore from semantic harmonics based on these examples.
* ❗️ After comparing the synthesized audio with the actual input audio of the guitar, it became evident that while the synthesized audio was "similar" to the actual guitar sound, **there were distinct differences in timbre**.
* ❗️ I thought it would be reasonable to examine the results by converting real mel spectrograms into audio to determine the source of the differences between the synthesized sound and the original guitar sound.
* ❗️I have confirmed that when converting the real guitar's mel spectrogram into audio, the timbre of the input audio is not reproduced accurately. **Therefore, in order to more accurately reproduce the timbre of the guitar, it is necessary to explore better methods for restoring the "phase information" of the guitar.** 
* ❗️Therefore, I need to find a better method than the GLA or discover better features than the Mel spectrogram.

<br>

## Outro
* What I obtained from this project was not an exact guitar playing sound but rather a sound resembling a guitar.  My model produces a sound similar to when I first started playing the guitar , with a gentle plucking sensation as if using my fingers instead of a pick.
* Of course, using more advanced generation algorithms could potentially achieve a sound closer to that of a guitar. To address this aspect, I need to continue exploring various approaches and strive to improve and grow in the future.
* **Anyway, I felt happy during this project because it made me feel like I fulfilled my childhood dream on my own. 😆**


<br>

#### +) To do: Key points I need to focus on
* The sound of a guitar is influenced by various factors.
  * The physical elements of the guitar itself: wood, strings, height of strings ...
  * The elements related to "playing": timbre when performing techniques such as sliding and hammering .... 
* It would be beneficial to first explore the characteristics in the frequency domain and also further investigate the impulse response of an acoustic guitar.
* Hyper-parameter optimization
* Attempts to approach using different generative model
* Mel to Audio inversion method (focusing on phase reconstruction) 


### Reference
> [1] Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs", in CVPR, 2018.  
> [2] Pix2PixHD official repository: https://github.com/NVIDIA/pix2pixHD/tree/master  
> [3] PyCeps: https://github.com/hwang9u/pyceps




### Cite

If you want to use this code, please cite as follows:
```
@misc{hwang9u-hum2guitar,
  author = {Kim Seonjoo},
  title = {hum2guitar},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hwang9u/hum2guitar}},
}
```


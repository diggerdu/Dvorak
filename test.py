import numpy as np
import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

fb = mfcc.get_filter_banks(filters_num = 26, NFFT = 512, samplerate = 8000,low_freq = 0, high_freq = 4000)

(rate , audio) = wav.read("bal.wav")
tmp, _ = mfcc.fbank(audio, samplerate = rate, filters_num = 26)

spectrum = mfcc.wav_spectrum_power(audio, samplerate = rate, filters_num = 26)
print spectrum.shape
print tmp.shape
print fb.shape

x = np.arange(257)
for y in fb:
    plt.plot(x,y,'b')
plt.show()

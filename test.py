import numpy as np
import mfcc



fb = mfcc.get_filter_banks(filters_num = 26, NFFT = 512, samplerate = 8000,low_freq = 0, high_freq = 4000)

print fb.shape

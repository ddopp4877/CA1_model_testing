import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from mpl_toolkits.mplot3d import Axes3D
import h5py
import matplotlib.pyplot as plt
from scipy.signal import welch
from bmtk.analyzer.compartment import plot_traces
import scipy.signal as ss
from scipy.signal import butter, lfilter, resample, filtfilt
from scipy.stats import zscore
from scipy.signal import freqz



tsim = 300
lfp_file = "output/ecp.h5"
f = h5py.File(lfp_file,'r')
lfp = list(f['ecp']['data'])
lfp_arr = np.asarray(lfp)
lfp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for i in range(13):
    lfp[i] = lfp_arr[:,i]
    lfp[i] = [(x*10)+i for x in lfp[i]]
    temp = lfp[i]
    temp = temp[1500:2500]
    plt.plot(np.arange(0,100,0.1),temp)
plt.xlabel('time (ms)')
plt.ylabel('channels')
plt.title("raw lfp for CA1")
plt.show()

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Filter requirements.
order = 6
fs = 10000       # sample rate, Hz
cutoff = 500  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
data=lfp[0]

y = butter_lowpass_filter(data, cutoff, fs, order)
filtered= resample(y, 1000)
filtered = zscore(filtered)

data = resample(data, 1000)
raw = zscore(data)

plt.plot(np.arange(0,1000, 1), filtered, label='filtered')
plt.plot(np.arange(0,1000, 1), raw, label='raw')

plt.legend()
plt.show()

lfp_file = "output/ecp.h5"
f = h5py.File(lfp_file,'r')
lfp = list(f['ecp']['data'])
lfp_arr = np.asarray(lfp)
lfp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for i in range(13):
    lfp[i] = lfp_arr[:,i]
    lfp[i] = [(x*10)+i for x in lfp[i]]
    y = butter_lowpass_filter(lfp[i], cutoff, fs, order)
    print(y.shape)
    filtered = resample(y, 1000)
    print(filtered.shape)
    filtered = filtered[500:804]
    plt.plot(np.arange(0, 100, 0.33), filtered)
plt.xlabel('time (ms)')
plt.ylabel('channel ')
plt.title("Filtered lfp for CA1")
plt.show()
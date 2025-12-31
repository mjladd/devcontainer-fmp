# %% [markdown]
# # Temporal Smoothing and Downsampling
#
# In this notebook, we discuss temporal smoothing and downsampling techniques for
# postprocessing feature representations. Parts of the notebook follow Section 3.1.2.3
# and Section 7.2.1 of [Müller, FMP, Springer 2015].

# %% [markdown]
# ## Beethoven Example
#
# To illustrate the effect of the various postprocessing techniques, we use the
# beginning of Beethoven's Fifth Symphony as our running example.

# %%
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import librosa
from scipy import signal

sys.path.append('..')
import libfmp.b
import libfmp.c3

%matplotlib inline

fn_wav_dict = {}
fn_wav_dict['Bernstein'] = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-21_Bernstein.wav')
fn_wav_dict['Karajan'] = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-21_Karajan1946.wav')
fn_wav_dict['Scherbakov'] = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-21_Scherbakov.wav')
fn_wav_dict['MIDI-Piano'] = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-21_Sibelius-Piano.wav')

N, H = 2048, 1024
figsize = (8, 1.5)
yticks = [0, 4, 7, 11]

C_dict = {}
for name in fn_wav_dict:
    fn_wav = fn_wav_dict[name]
    x, Fs = librosa.load(fn_wav)
    C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
    Fs_C = Fs / H
    C = C / C.max()
    threshold = 0.0001
    C_dict[name] = libfmp.c3.normalize_feature_sequence(C, norm='2', threshold=threshold)
    libfmp.b.plot_chromagram(C_dict[name], Fs_C, figsize=figsize, ylabel=name, xlabel='',
                             chroma_yticks=yticks)

# %% [markdown]
# ## Temporal Smoothing and Downsampling
#
# The idea is to compute for each chroma dimension a kind of local average over time.
# Let X = (x_1, x_2, ..., x_N) be a feature sequence, and let w be a rectangular
# window of length L. Then we compute for each k a convolution between w and the
# sequence (x_1(k), x_2(k), ..., x_N(k)).

# %%
def smooth_downsample_feature_sequence(X, Fs, filt_len=41, down_sampling=10, w_type='boxcar'):
    """Smoothes and downsamples a feature sequence. Smoothing is achieved by
    convolution with a filter kernel

    Args:
        X (np.ndarray): Feature sequence
        Fs (scalar): Frame rate of X
        filt_len (int): Length of smoothing filter
        down_sampling (int): Downsampling factor
        w_type (str): Window type of smoothing filter

    Returns:
        X_smooth (np.ndarray): Smoothed and downsampled feature sequence
        Fs_feature (scalar): Frame rate of X_smooth
    """
    filt_kernel = np.expand_dims(signal.get_window(w_type, filt_len), axis=0)
    X_smooth = signal.convolve(X, filt_kernel, mode='same') / filt_len
    X_smooth = X_smooth[:, ::down_sampling]
    Fs_feature = Fs / down_sampling
    return X_smooth, Fs_feature


filt_len = 11
down_sampling = 2
C_smooth_dict = {}
for name in fn_wav_dict:
    C_smooth, Fs_C_smooth = smooth_downsample_feature_sequence(C_dict[name], Fs_C,
                                        filt_len=filt_len, down_sampling=down_sampling)
    C_smooth_dict[name] = libfmp.c3.normalize_feature_sequence(C_smooth, norm='2', threshold=threshold)
    libfmp.b.plot_chromagram(C_smooth_dict[name], Fs_C_smooth, figsize=figsize,
                             ylabel=name, title='', xlabel='', chroma_yticks=yticks)

# %% [markdown]
# ## Smoothing via Median Filtering
#
# An alternative to applying a local averaging filter is **median filtering**,
# which also results in some smoothing while better preserving sharp transitions.

# %%
X = np.array([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9], [5, 3, 2, 8, 2]], dtype='float')
L = 3
filt_len = [1, L]
X_smooth = signal.medfilt2d(X, filt_len)
print('Input array X of dimension (K,N) with K=3 and N=5')
print(X)
print('Output array after median filtering with L=3')
print(X_smooth)

# %%
def median_downsample_feature_sequence(X, Fs, filt_len=41, down_sampling=10):
    """Smoothes and downsamples a feature sequence. Smoothing is achieved by
    median filtering

    Args:
        X (np.ndarray): Feature sequence
        Fs (scalar): Frame rate of X
        filt_len (int): Length of smoothing filter
        down_sampling (int): Downsampling factor

    Returns:
        X_smooth (np.ndarray): Smoothed and downsampled feature sequence
        Fs_feature (scalar): Frame rate of X_smooth
    """
    assert filt_len % 2 == 1  # L needs to be odd
    filt_len = [1, filt_len]
    X_smooth = signal.medfilt2d(X, filt_len)
    X_smooth = X_smooth[:, ::down_sampling]
    Fs_feature = Fs / down_sampling
    return X_smooth, Fs_feature


filt_len = 11
down_sampling = 2
C_median_dict = {}
for name in fn_wav_dict:
    C_median, Fs_C_smooth = median_downsample_feature_sequence(C_dict[name], Fs_C,
                                                               filt_len=filt_len, down_sampling=down_sampling)
    C_median_dict[name] = libfmp.c3.normalize_feature_sequence(C_median, norm='2', threshold=threshold)

figsize = (8, 1.7)

name = 'Karajan'
libfmp.b.plot_chromagram(C_dict[name], Fs_C_smooth, figsize=figsize, ylabel=name,
                         title='Original chromagram', xlabel='', chroma_yticks=yticks)
libfmp.b.plot_chromagram(C_smooth_dict[name], Fs_C_smooth, figsize=figsize, ylabel=name,
                         title='Smoothed chromagram using average filtering', xlabel='', chroma_yticks=yticks)
libfmp.b.plot_chromagram(C_median_dict[name], Fs_C_smooth, figsize=figsize, ylabel=name,
                         title='Smoothed chromagram using median filtering', xlabel='', chroma_yticks=yticks)

C_diff = C_smooth_dict[name] - C_median_dict[name]
m = np.max(np.abs(C_diff))
libfmp.b.plot_chromagram(C_diff, Fs_C_smooth, cmap='seismic', clim=[-m, m], figsize=figsize,
                         title='Difference between average- and median-filtered chromagram', xlabel='',
                         ylabel=name, chroma_yticks=yticks)

# %% [markdown]
# ## Further Notes
#
# In summary, there are many ways to enhance and modify a feature representation
# by applying techniques such as logarithmic compression, feature normalization,
# smoothing, and downsampling. The described techniques provide flexible and
# computationally inexpensive tools for adjusting the feature specificity and
# resolution without repeating the cost-intensive spectral audio decomposition.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Vlora Arifi-Müller.

# %% [markdown]
# # SSM: Feature Smoothing
#
# In this notebook, we study the effect of feature smoothing on structural properties
# of the resulting SSM. Parts of this notebook follow Section 4.2.2.1 of
# [Müller, FMP, Springer 2015].

# %% [markdown]
# ## Example: Brahms
#
# When computing an SSM, the given waveform-based audio recording is first transformed
# into a suitable feature representation, which captures specific acoustic and musical
# properties. In this notebook, we study the influence of feature smoothing and
# downsampling on an SSM.

# %%
import numpy as np
import os, sys, librosa
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import IPython.display as ipd
import pandas as pd
from numba import jit


sys.path.append('..')
import libfmp.b
from libfmp.b import FloatingBox
import libfmp.c2
import libfmp.c3
import libfmp.c4

%matplotlib inline

# %% [markdown]
# ## Moderate Smoothing
#
# Starting with normalized chroma features with a feature rate of 10 Hz, the resulting
# SSM yields a very detailed description of repetitive structures. Using smoothing and
# downsampling, one obtains an SSM where many of the details have been smoothed out,
# and some of the structurally relevant path and block structures have become more
# prominent.

# %%
# Annotation
filename = 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.csv'
fn_ann = os.path.join('..', 'data', 'C4', filename)
ann, color_ann = libfmp.c4.read_structure_annotation(fn_ann, fn_ann_color=filename)

# Waveform
fn_wav = os.path.join('..', 'data', 'C4', 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)
x_duration = (x.shape[0])/Fs

# Chroma Feature Sequence and SSM (10 Hz)
C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=2, hop_length=2205, n_fft=4410)
Fs_C = Fs/2205

# Chroma Feature Sequence and SSM (10 Hz)
L, H = 1, 1
X, Fs_feature = libfmp.c3.smooth_downsample_feature_sequence(C, Fs_C,
                        filt_len=L, down_sampling=H)
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
S = libfmp.c4.compute_sm_dot(X,X)
ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_feature)
fig, ax = libfmp.c4.plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_feature,
            label='Time (frames)', color_ann=color_ann, clim=[0,1], clim_X=[0,1],
            title='Feature rate = %0.0f Hz (L = %d, H = %d)'%(Fs_feature,L,H))
float_box = libfmp.b.FloatingBox()
float_box.add_fig(fig)

# Chroma Feature Sequence and SSM (10 Hz)
L, H = 41, 10
X, Fs_feature = libfmp.c3.smooth_downsample_feature_sequence(C, Fs_C,
                        filt_len=L, down_sampling=H)
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
S = libfmp.c4.compute_sm_dot(X,X)
ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_feature)
fig, ax = libfmp.c4.plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_feature,
            label='Time (frames)', color_ann=color_ann, clim=[0,1], clim_X=[0,1],
            title='Feature rate = %0.0f Hz (L = %d, H = %d)'%(Fs_feature,L,H))
float_box.add_fig(fig)

float_box.show()

# %% [markdown]
# ## Strong Smoothing
#
# Further increasing the smoothing length and reducing the feature rate results in an
# emphasis of the rough harmonic content. Using large smoothing windows, relevant path
# structures may be smeared out and lost for the subsequent steps. For other applications
# such as homogeneity-based structure analysis, however, averaging over large windows
# may be beneficial.

# %%
# Chroma Feature Sequence and SSM (0.5 Hz)
L, H = 161, 20
X, Fs_feature = libfmp.c3.smooth_downsample_feature_sequence(C, Fs_C,
                        filt_len=L, down_sampling=H)
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
S = libfmp.c4.compute_sm_dot(X,X)
ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_feature)
fig, ax = libfmp.c4.plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_feature,
            label='Time (frames)', color_ann=color_ann, clim=[0,1], clim_X=[0,1],
            title='Feature rate = %0.2f Hz (L = %d, H = %d)'%(Fs_feature,L,H))
float_box = libfmp.b.FloatingBox()
float_box.add_fig(fig)

# Chroma Feature Sequence and SSM (0.25 Hz)
L, H = 321, 40
X, Fs_feature = libfmp.c3.smooth_downsample_feature_sequence(C, Fs_C,
                        filt_len=L, down_sampling=H)
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
S = libfmp.c4.compute_sm_dot(X,X)
ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_feature)
fig, ax = libfmp.c4.plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_feature,
            label='Time (frames)', color_ann=color_ann, clim=[0,1], clim_X=[0,1],
            title='Feature rate = %0.2f Hz (L = %d, H = %d)'%(Fs_feature,L,H))
float_box.add_fig(fig)

float_box.show()

# %% [markdown]
# ## Median Filtering
#
# **Median filtering** is an alternative to average filtering. Recall that median
# filtering tends to better preserve edges or sharp transient between homogeneous
# regions.

# %%
# Chroma Feature Sequence and SSM (0.5 Hz)
L_iter = [11, 31, 91, 271]
H_iter = [ 3, 9, 27, 81]
num_iter = len(L_iter)

print('SSMs obtained using average filtering')
fig = plt.figure(figsize=(10,3))
for i in range(num_iter):
    L = L_iter[i]
    H = H_iter[i]
    X, Fs_feature = libfmp.c3.smooth_downsample_feature_sequence(C, Fs_C,
                        filt_len=L, down_sampling=H)
    X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
    S = libfmp.c4.compute_sm_dot(X,X)
    ax = fig.add_subplot(1, num_iter, i+1)
    im = plt.imshow(S, cmap='gray_r', aspect='equal', origin='lower')
    ax.title.set_text('L = %d, H = %d'%(L,H))
plt.tight_layout()
plt.show()

print('SSMs obtained using median filtering')
fig = plt.figure(figsize=(10,3))
for i in range(num_iter):
    L = L_iter[i]
    H = H_iter[i]
    X, Fs_feature = libfmp.c3.median_downsample_feature_sequence(C, Fs_C,
                        filt_len=L, down_sampling=H)
    X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
    S = libfmp.c4.compute_sm_dot(X,X)
    ax = fig.add_subplot(1, num_iter, i+1)
    im = plt.imshow(S, cmap='gray_r', aspect='equal', origin='lower')
    ax.title.set_text('L = %d, H = %d'%(L,H))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Further Notes
#
# In this notebook, we discussed the importance of the size of the analysis window and
# the feature rate. Knowing the temporal level of the music processing task is of great
# help for choosing suitable parameters. For tasks such as extracting the musical
# structure from a given audio recording, smoothing and downsampling already on the
# feature level can lead to substantial improvements. Another important strategy for
# adjusting and reducing the feature rate is based on **adaptive windowing**, where
# the analysis windows are determined by previously extracted onset and beat positions.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

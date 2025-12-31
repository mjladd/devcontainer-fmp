# %% [markdown]
# # Self-Similarity Matrix (SSM)
#
# To study musical structures and their mutual relations, one general idea is to convert
# the music signal into a suitable feature sequence and then to compare each element of
# the feature sequence with all other elements of the sequence. This results in a
# **self-similarity matrix** (SSM), a tool which is of fundamental importance not only
# for music structure analysis but also for the analysis of many kinds of time series.
# Following Section 4.2.1 of [Müller, FMP, Springer 2015], we introduce in this notebook
# the concept of SSMs and discuss their fundamental properties.

# %% [markdown]
# ## Basic Definition
#
# Let F be feature space and s:F×F→R a similarity measure that makes it possible to
# compare two elements x,y∈F. Given a feature sequence X=(x_1,x_2,...,x_N), the idea
# is to compare all elements of the sequence with each other. This results in an N-square
# **self-similarity matrix** S∈R^(N×N) defined by S(n,m):=s(x_n,x_m).
#
# A simple similarity measure s is the inner product defined by s(x,y) := <x,y>
# for two vectors x,y∈F. With this similarity measure, the score between two orthogonal
# feature vectors is zero and otherwise it is nonzero.

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
import libfmp.c2
import libfmp.c3
import libfmp.c4
import libfmp.c6

%matplotlib inline

# Generate normalized feature sequence
K = 4
M = 100
r = np.arange(M)
b1 = np.zeros((K,M))
b1[0,:] = r
b1[1,:] = M-r
b2 = np.ones((K,M))
X = np.concatenate(( b1, b1, np.roll(b1, 2, axis=0), b2, b1 ), axis=1)
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)

# Compute SSM
S = np.dot(np.transpose(X), X)

# Visualization
cmap = 'gray_r'
fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                          'height_ratios': [0.2, 1]}, figsize=(4.5, 5))
libfmp.b.plot_matrix(X, Fs=1, ax=[ax[0,0], ax[0,1]], cmap=cmap,
            xlabel='Time (frames)', ylabel='', title='Feature sequence')
libfmp.b.plot_matrix(S, Fs=1, ax=[ax[1,0], ax[1,1]], cmap=cmap,
            title='SSM', xlabel='Time (frames)', ylabel='Time (frames)', colorbar=True);
plt.tight_layout()

# %%
# The visual appearance can be changed by suitable adjusting the colormap
cmap = libfmp.b.compressed_gray_cmap(alpha=-1000)
fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                          'height_ratios': [0.2, 1]}, figsize=(4.5, 5))
libfmp.b.plot_matrix(X, Fs=1, ax=[ax[0,0], ax[0,1]], cmap=cmap,
            xlabel='Time (frames)', ylabel='', title='Feature sequence')
libfmp.b.plot_matrix(S, Fs=1, ax=[ax[1,0], ax[1,1]], cmap=cmap,
            title='SSM', xlabel='Time (frames)', ylabel='Time (frames)', colorbar=True);
plt.tight_layout()

# %% [markdown]
# ## Block and Path Structures
#
# The two most prominent structures in SSMs are referred to as **blocks** and **paths**.
# If the feature sequence captures musical properties that stay somewhat constant over
# the duration of an entire musical part, each of the feature vectors is similar to all
# other feature vectors within this segment. As a result, an entire **block** of large
# values appears in the SSM. In other words, **homogeneity properties correspond to
# block-like structures**. If the feature sequence contains two repeating subsequences,
# the corresponding elements of the two subsequences are similar to each other. As a
# result, a **path** (or **stripe**) of high similarity running parallel to the main
# diagonal becomes visible in the SSM. In other words, **repetitive properties correspond
# to path-like structures**.

# %% [markdown]
# ## SSM Based on Chromagram Features

# %%
@jit(nopython=True)
def compute_sm_dot(X, Y):
    """Computes similarty matrix from feature sequences using dot (inner) product

    Notebook: C4/C4S2_SSM.ipynb

    Args:
        X (np.ndarray): First sequence
        Y (np.ndarray): Second Sequence

    Returns:
        S (float): Dot product
    """
    S = np.dot(np.transpose(X), Y)
    return S

def plot_feature_ssm(X, Fs_X, S, Fs_S, ann, duration, color_ann=None,
                     title='', label='Time (seconds)', time=True,
                     figsize=(5, 6), fontsize=10, clim_X=None, clim=None):
    """Plot SSM along with feature representation and annotations

    Notebook: C4/C4S2_SSM.ipynb

    Args:
        X: Feature representation
        Fs_X: Feature rate of ``X``
        S: Similarity matrix (SM)
        Fs_S: Feature rate of ``S``
        ann: Annotaions
        duration: Duration
        color_ann: Color annotations (Default value = None)
        title: Figure title (Default value = '')
        label: Label for time axes (Default value = 'Time (seconds)')
        time: Display time axis ticks or not (Default value = True)
        figsize: Figure size (Default value = (5, 6))
        fontsize: Font size (Default value = 10)
        clim_X: Color limits for matrix X (Default value = None)
        clim: Color limits for matrix ``S`` (Default value = None)

    Returns:
        fig: Handle for figure
        ax: Handle for axes
    """
    cmap = libfmp.b.compressed_gray_cmap(alpha=-10)
    fig, ax = plt.subplots(3, 3, gridspec_kw={'width_ratios': [0.1, 1, 0.05],
                                              'wspace': 0.2,
                                              'height_ratios': [0.3, 1, 0.1]},
                           figsize=figsize)
    libfmp.b.plot_matrix(X, Fs=Fs_X, ax=[ax[0, 1], ax[0, 2]], clim=clim_X,
                         xlabel='', ylabel='', title=title)
    ax[0, 0].axis('off')
    libfmp.b.plot_matrix(S, Fs=Fs_S, ax=[ax[1, 1], ax[1, 2]], cmap=cmap, clim=clim,
                         title='', xlabel='', ylabel='', colorbar=True)
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    libfmp.b.plot_segments(ann, ax=ax[2, 1], time_axis=time, fontsize=fontsize,
                           colors=color_ann,
                           time_label=label, time_max=duration*Fs_X)
    ax[2, 2].axis('off')
    ax[2, 0].axis('off')
    libfmp.b.plot_segments(ann, ax=ax[1, 0], time_axis=time, fontsize=fontsize,
                           direction='vertical', colors=color_ann,
                           time_label=label, time_max=duration*Fs_X)
    return fig, ax

# Waveform
fn_wav = os.path.join('..', 'data', 'C4', 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)
x_duration = (x.shape[0])/Fs

# Chroma Feature Sequence
N, H = 4096, 1024
chromagram = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=2, hop_length=H, n_fft=N)
X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(chromagram, Fs/H, filt_len=41, down_sampling=10)

# Annotation
filename = 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.csv'
fn_ann = os.path.join('..', 'data', 'C4', filename)
ann, color_ann = libfmp.c4.read_structure_annotation(fn_ann, fn_ann_color=filename)
ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X)

# SSM
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
S = compute_sm_dot(X,X)
fig, ax = plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_X, color_ann=color_ann,
                           clim_X=[0,1], clim=[0,1], label='Time (frames)',
                           title='Chroma feature (Fs=%0.2f)'%Fs_X)

# %% [markdown]
# ## SSM Based on MFCC Features
#
# MFCC-based features are a mid-level representation that somehow correlate to aspects
# such as **instrumentation** and **timbre**.

# %%
from libfmp.b import FloatingBox
float_box = libfmp.b.FloatingBox()

# MFCC-based feature sequence
N, H = 2048, 1024
X_MFCC = librosa.feature.mfcc(y=x, sr=Fs, hop_length=H, n_fft=N)
coef = np.arange(0,20)
X_MFCC_upper = X_MFCC[coef,:]
X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(X_MFCC_upper, Fs/H, filt_len=41, down_sampling=10)
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
S = compute_sm_dot(X,X)
ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X)
fig, ax = plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_X, color_ann=color_ann,
    title='MFCC (20 coefficients, Fs=%0.2f)'%Fs_X, label='Time (frames)')
float_box.add_fig(fig)


# MFCC-based feature sequence only using coefficients 4 to 14
coef = np.arange(4,15)
X_MFCC_upper = X_MFCC[coef,:]
X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(X_MFCC_upper, Fs/H, filt_len=41, down_sampling=10)
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
S = compute_sm_dot(X,X)
ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X)
fig, ax = plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_X,
                           color_ann=color_ann, label='Time (frames)',
                           title='MFCC (coefficients 4 to 14, Fs=%0.2f)'%Fs_X)
float_box.add_fig(fig)

float_box.show()

# %% [markdown]
# ## SSM Based on Tempogram Features

# %%
# Tempogram feature sequence
nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512, gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

N, H = 1000, 100
X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs_nov, N=N, H=H, Theta=np.arange(30, 601))
octave_bin = 12
tempogram_F = np.abs(X)
output = libfmp.c6.compute_cyclic_tempogram(tempogram_F, F_coef_BPM, octave_bin=octave_bin)
X = output[0]
F_coef_scale = output[1]
Fs_X = Fs_nov/H
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
S = compute_sm_dot(X,X)
ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X)
fig, ax = plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_X, color_ann=color_ann,
    title='Tempogram (Fs=%0.2f)'%Fs_X, label='Time (frames)')

# %% [markdown]
# ## Overall Procedure for Music Structure Analysis
#
# Most computational approaches to music structure analysis exploit path- and block-like
# structures of SSMs in one way or another, and the overall algorithmic pipelines
# typically contain the following general steps:
#
# * The music signal is transformed into a suitable feature sequence.
# * A self-similarity matrix is computed from the feature sequence based on a similarity measure.
# * Blocks and paths of high overall score are derived from the SSM. Each block or path
#   defines a pair of similar segments.
# * Entire groups of mutually similar segments are formed from the pairwise relations
#   by applying a clustering step.
#
# The last step can be considered as forming a kind of transitive closure of the pairwise
# segment relations induced by block and path structures.

# %% [markdown]
# ## Further Notes
#
# In the subsequent notebooks, we will cover the following enhancement and postprocessing
# strategies:
#
# * Feature smoothing
# * Path enhancement
# * Transposition invariance
# * Thresholding

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

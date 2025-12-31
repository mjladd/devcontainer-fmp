# %% [markdown]
# # SSM: Transposition Invariance
#
# Following Section 4.2.2.3 of [Müller, FMP, Springer 2015], we introduce in this
# notebook the concept of a transposition-invariant SSM.

# %% [markdown]
# ## Example: In the Year 2525 (Zager and Evans)
#
# It is often the case that certain musical parts are repeated in a transposed form,
# where the melody is moved up or down in pitch by a constant interval. As a motivating
# example, let us consider the song "In the Year 2525" by Zager and Evans. The song has
# the overall musical structure IV1V2V3V4V5V6V7BV8O. While the first four verse sections
# are in the same musical key, V5 and V6 are transposed by one semitone upwards, and V7
# and V8 are transposed by two semitones upwards.

# %%
import numpy as np
import os, sys, librosa
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import IPython.display as ipd
import pandas as pd
from numba import jit

sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c3
import libfmp.c4

%matplotlib inline

# Annotation
filename = 'FMP_C4_F13_ZagerEvans_InTheYear2525.csv'
fn_ann = os.path.join('..', 'data', 'C4', filename)
ann, color_ann = libfmp.c4.read_structure_annotation(fn_ann, fn_ann_color=filename)

# Waveform
fn_wav = os.path.join('..', 'data', 'C4', 'FMP_C4_F13_ZagerEvans_InTheYear2525.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)
x_duration = (x.shape[0])/Fs

# Chroma Feature Sequence and SSM (10 Hz)
C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=2, hop_length=2205, n_fft=4410)
Fs_C = Fs/2205
X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(C, Fs_C, filt_len=41, down_sampling=10)

X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
S = libfmp.c4.compute_sm_dot(X,X)
L = 8
tempo_rel_set = np.array([1])
S_forward = libfmp.c4.filter_diag_mult_sm(S, L, tempo_rel_set=tempo_rel_set, direction=0)
S_backward = libfmp.c4.filter_diag_mult_sm(S, L, tempo_rel_set=tempo_rel_set, direction=1)
S_final = np.maximum(S_forward, S_backward)
ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X)
fig, ax = libfmp.c4.plot_feature_ssm(X, 1, S_final, 1, ann_frames, x_duration*Fs_X,
            label='Time (frames)', color_ann=color_ann, fontsize=9, clim_X=[0,1], clim=[0.5,1],
            title='Feature rate: %0.0f Hz'%(Fs_X))

# %% [markdown]
# ## Cyclic Shift Operator
#
# Transpositions can be simulated by cyclically shifting chroma features. Identifying
# the twelve chroma values with the set [0:11], a cyclic shift is modeled by the cyclic
# shift operator.

# %%
@jit(nopython=True)
def shift_cyc_matrix(X, shift=0):
    """Cyclic shift of features matrix along first dimension

    Notebook: C4/C4S2_SSM-TranspositionInvariance.ipynb

    Args:
        X (np.ndarray): Feature respresentation
        shift (int): Number of bins to be shifted (Default value = 0)

    Returns:
        X_cyc (np.ndarray): Cyclically shifted feature matrix
    """
    K, N = X.shape
    shift = np.mod(shift, K)
    X_cyc = np.zeros((K, N))
    X_cyc[shift:K, :] = X[0:K-shift, :]
    X_cyc[0:shift, :] = X[K-shift:K, :]
    return X_cyc

shift_set = [0,1,2]
shift_num = len(shift_set)

hr = np.ones(shift_num+1)
hr[-1] = 0.4
fig, ax = plt.subplots(shift_num+1, 2, gridspec_kw={'width_ratios': [1, 0.02],
                                                    'height_ratios': hr}, figsize=(6, 6))

for m in range(shift_num):
    shift = shift_set[m]
    X_cyc = shift_cyc_matrix(X, shift)
    fig_im, ax_im, im = libfmp.b.plot_matrix(X_cyc, Fs=Fs_X, ax=[ax[m,0], ax[m,1]],
                     title=r'$%d$-transposed chromgram'%shift, ylabel='Chroma', colorbar=True);
    libfmp.b.plot_segments_overlay(ann, ax=ax_im[0], time_max=(x.shape[0])/Fs,
                                   print_labels=False, label_ticks=False,
                                   colors=color_ann, fontsize=10, alpha=0.2)

libfmp.b.plot_segments(ann, ax=ax[shift_num,0], time_max=(x.shape[0])/Fs, colors=color_ann, fontsize=12)
ax[shift_num,0].set_xlabel('Time (seconds)')
ax[shift_num,1].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Transposed SSM
#
# For a given feature sequence X, we define the i-transposed self-similarity matrix.

# %%
L = 8
tempo_rel_set = np.asarray([1])
shift_set = np.asarray([0,1,2])
shift_num = len(shift_set)
fig, ax = plt.subplots(1, shift_num, figsize=(10, 3))
for m in range(shift_num):
    shift = shift_set[m]
    X_cyc = shift_cyc_matrix(X, shift)
    S = libfmp.c4.compute_sm_dot(X,X_cyc)
    S_forward = libfmp.c4.filter_diag_mult_sm(S, L, tempo_rel_set=tempo_rel_set, direction=0)
    S_backward = libfmp.c4.filter_diag_mult_sm(S, L, tempo_rel_set=tempo_rel_set, direction=1)
    S_final = np.maximum(S_forward, S_backward)
    libfmp.c4.subplot_matrix_colorbar(S_final, fig, ax[m], clim=[0.5,1],
                        title=r'$%5d$-transposed SSM'%shift)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Transposition-Invariant SSM
#
# Taking a cell-wise maximum over the twelve different cyclic shifts, we obtain a single
# transposition-invariant self-similarity matrix.

# %%
def compute_sm_ti(X, Y, L=1, tempo_rel_set=np.asarray([1]), shift_set=np.asarray([0]), direction=2):
    """Compute enhanced similaity matrix by applying path smoothing and transpositions

    Notebook: C4/C4S2_SSM-TranspositionInvariance.ipynb

    Args:
        X (np.ndarray): First feature sequence
        Y (np.ndarray): Second feature sequence
        L (int): Length of filter (Default value = 1)
        tempo_rel_set (np.ndarray): Set of relative tempo values (Default value = np.asarray([1]))
        shift_set (np.ndarray): Set of shift indices (Default value = np.asarray([0]))
        direction (int): Direction of smoothing (0: forward; 1: backward; 2: both) (Default value = 2)

    Returns:
        S_TI (np.ndarray): Transposition-invariant SM
        I_TI (np.ndarray): Transposition index matrix
    """
    for shift in shift_set:
        Y_cyc = shift_cyc_matrix(Y, shift)
        S_cyc = libfmp.c4.compute_sm_dot(X, Y_cyc)

        if direction == 0:
            S_cyc = libfmp.c4.filter_diag_mult_sm(S_cyc, L, tempo_rel_set, direction=0)
        if direction == 1:
            S_cyc = libfmp.c4.filter_diag_mult_sm(S_cyc, L, tempo_rel_set, direction=1)
        if direction == 2:
            S_forward = libfmp.c4.filter_diag_mult_sm(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=0)
            S_backward = libfmp.c4.filter_diag_mult_sm(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=1)
            S_cyc = np.maximum(S_forward, S_backward)
        if shift == shift_set[0]:
            S_TI = S_cyc
            I_TI = np.ones((S_cyc.shape[0], S_cyc.shape[1])) * shift
        else:
            I_TI[S_cyc > S_TI] = shift
            S_TI = np.maximum(S_cyc, S_TI)

    return S_TI, I_TI

def subplot_matrix_ti_colorbar(S, fig, ax, title='', Fs=1, xlabel='Time (seconds)', ylabel='Time (seconds)',
                               clim=None, xlim=None, ylim=None, cmap=None, alpha=1, interpolation='nearest',
                               ind_zero=False):
    """Visualization function for showing transposition index matrix"""
    if cmap is None:
        color_ind_zero = np.array([0, 0, 0, 1])
        if ind_zero == 0:
            color_ind_zero = np.array([0, 0, 0, 1])
        else:
            color_ind_zero = np.array([1, 1, 1, 1])
        colorList = np.array([color_ind_zero, [1, 1, 0, 1],  [0, 0.7, 0, 1],  [1, 0, 1, 1],  [0, 0, 1, 1],
                             [1, 0, 0, 1], [0, 0, 0, 0.5], [1, 0, 0, 0.3], [0, 0, 1, 0.3], [1, 0, 1, 0.3],
                             [0, 0.7, 0, 0.3], [1, 1, 0, 0.3]])
        cmap = ListedColormap(colorList)
    len_sec = S.shape[0] / Fs
    extent = [0, len_sec, 0, len_sec]
    im = ax.imshow(S, aspect='auto', extent=extent, cmap=cmap, origin='lower', alpha=alpha,
                   interpolation=interpolation)
    if clim is None:
        im.set_clim(vmin=-0.5, vmax=11.5)
    fig.sca(ax)
    ax_cb = fig.colorbar(im)
    ax_cb.set_ticks(np.arange(0, 12, 1))
    ax_cb.set_ticklabels(np.arange(0, 12, 1))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return im

L = 8
tempo_rel_set = np.asarray([1])
shift_set = np.array(range(12))
S_TI, I_TI = compute_sm_ti(X, X, L=L, tempo_rel_set=tempo_rel_set,
                           shift_set=shift_set, direction=2)

fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
libfmp.c4.subplot_matrix_colorbar(S_TI, fig, ax[0], clim=[0.5,1],
                                  title='Transposition-invariant SSM')
subplot_matrix_ti_colorbar(I_TI, fig, ax[1], ind_zero=True,
                                  title='Transposition index matrix')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Dependency on Parameters
#
# The following figure shows that the choice of parameters have a crucial impact on
# the final result.

# %%
C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=2, hop_length=2205, n_fft=4410)
Fs_C = Fs/2205

L_feature = 41
H_feature = 10
X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(C, Fs_C,
                                    filt_len=L_feature, down_sampling=H_feature)
X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)

tempo_rel_min = 0.66
tempo_rel_max = 1.5
num = 5
tempo_rel_set = libfmp.c4.compute_tempo_rel_set(tempo_rel_min=tempo_rel_min, tempo_rel_max=tempo_rel_max, num=num)

shift_set = np.array(range(12))

L_set = [1, 20]
L_num = len(L_set)
title_set = ['Transposition-invariant SSM', 'Smoothed transposition-invariant SSM']

fig, ax = plt.subplots(L_num, 2, figsize=(8, 7))
for m in range(L_num):
    L = L_set[m]
    S_TI, I_TI = compute_sm_ti(X, X, L=L, tempo_rel_set=tempo_rel_set, shift_set=shift_set, direction=2)
    libfmp.c4.subplot_matrix_colorbar(S_TI, fig, ax[m,0], clim=[0.5,1],
                                  title=title_set[m])
    subplot_matrix_ti_colorbar(I_TI, fig, ax[m,1], ind_zero=True,
                                  title='Transposition index matrix')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and David Kopyto.

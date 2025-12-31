# %% [markdown]
# # SSM: Path Enhancement
#
# Following Section 4.2.2.2 of [Müller, FMP, Springer 2015], we discuss in this notebook
# a strategy for enhancing path structures in SSMs.

# %% [markdown]
# ## Example: Brahms
#
# As our running example in this notebook, we consider again the Ormandy recording of
# the Hungarian Dance No. 5 by Johannes Brahms. As a feature representation, we use
# a chromagram with a feature resolution of 2 Hz.

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

%matplotlib inline

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

# Chroma Feature Sequence and SSM (2 Hz)
L, H = 21, 5
X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(C, Fs_C,
                        filt_len=L, down_sampling=H)

X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
S = libfmp.c4.compute_sm_dot(X,X)
ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X)
fig, ax = libfmp.c4.plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_X,
            label='Time (frames)', color_ann=color_ann, clim=[0,1], clim_X=[0,1],
            title='Feature rate = %0.0f Hz (L = %d, H = %d)'%(Fs_X,L,H))

# %% [markdown]
# ## Diagonal Smoothing
#
# One important property of similarity matrices is the appearance of paths of high
# similarity that are parallel to the main diagonal. To enhance the path structure,
# one general strategy is to apply some kind of smoothing filter along the direction
# of the main diagonal, resulting in an emphasis of diagonal information in S and a
# denoising of other structures. Such a filtering process is referred to as
# **diagonal smoothing**.

# %%
@jit(nopython=True)
def filter_diag_sm(S, L):
    """Path smoothing of similarity matrix by forward filtering along main diagonal

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        S (np.ndarray): Similarity matrix (SM)
        L (int): Length of filter

    Returns:
        S_L (np.ndarray): Smoothed SM
    """
    N = S.shape[0]
    M = S.shape[1]
    S_L = np.zeros((N, M))
    S_extend_L = np.zeros((N + L, M + L))
    S_extend_L[0:N, 0:M] = S
    for pos in range(0, L):
        S_L = S_L + S_extend_L[pos:(N + pos), pos:(M + pos)]
    S_L = S_L / L
    return S_L

def subplot_matrix_colorbar(S, fig, ax, title='', Fs=1,
                            xlabel='Time (seconds)', ylabel='Time (seconds)',
                            clim=None, xlim=None, ylim=None, cmap=None, interpolation='nearest'):
    """Visualization function for showing zoomed sections of matrices

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        S: Similarity matrix (SM)
        fig: Figure handle
        ax: Axes handle
        title: Title for figure (Default value = '')
        Fs: Feature rate (Default value = 1)
        xlabel: Label for x-axis (Default value = 'Time (seconds)')
        ylabel: Label for y-axis (Default value = 'Time (seconds)')
        clim: Color limits (Default value = None)
        xlim: Limits for x-axis (Default value = None)
        ylim: Limits for x-axis (Default value = None)
        cmap: Colormap for imshow (Default value = None)
        interpolation: Interpolation value for imshow (Default value = 'nearest')

    Returns:
        im: Imshow handle
    """
    if cmap is None:
        cmap = libfmp.b.compressed_gray_cmap(alpha=-100)
    len_sec = S.shape[0] / Fs
    extent = [0, len_sec, 0, len_sec]
    im = ax.imshow(S, aspect='auto', extent=extent, cmap=cmap,  origin='lower', interpolation=interpolation)
    fig.sca(ax)
    fig.colorbar(im)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if clim is not None:
        im.set_clim(clim)
    return im

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
subplot_matrix_colorbar(S, fig, ax[0], clim=[0,1], ylabel='Time (frames)', xlabel='Time (frames)',
                     title=r'Original SSM $\mathbf{S}$')
L = 20
S_L = filter_diag_sm(S, L)
subplot_matrix_colorbar(S_L, fig, ax[1], clim=[0,1], ylabel='Time (frames)', xlabel='Time (frames)',
                     title=r'Smoothed SSM $\mathbf{S}_{L}$ (L = %d)'%L)
L_long = 40
S_L_long = filter_diag_sm(S, L_long)
subplot_matrix_colorbar(S_L_long, fig, ax[2], clim=[0,1], ylabel='Time (frames)', xlabel='Time (frames)',
                     title=r'Smoothed SSM $\mathbf{S}_{L}$ (L = %d)'%L_long)

plt.tight_layout()
plt.show()

# %% [markdown]
# A simple filtering along the main diagonal works well if there are no relative tempo
# differences between the segments to be compared. However, this assumption is violated
# when a part is repeated with a faster or slower tempo.

# %%
xlim = [75, 150]
ylim = [125, 200]

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
subplot_matrix_colorbar(S, fig, ax[0], clim=[0,1], xlim=xlim, ylim=ylim,
                        title=r'Original SSM $\mathbf{S}$',
                        xlabel='Time (frames)', ylabel='Time (frames)')
subplot_matrix_colorbar(S_L, fig, ax[1], clim=[0,1], xlim=xlim, ylim=ylim,
                        title=r'Smoothed SSM $\mathbf{S}_{L}$ (L = %d)'%L,
                        xlabel='Time (frames)', ylabel='Time (frames)')
subplot_matrix_colorbar(S_L_long, fig, ax[2], clim=[0,1], xlim=xlim, ylim=ylim,
                        title=r'Smoothed SSM $\mathbf{S}_{L}$ (L = %d)'%L_long,
                        xlabel='Time (frames)', ylabel='Time (frames)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Multiple Filtering
#
# To deal with non-diagonal path structures due relative tempo differences, we now
# introduce a **multiple filtering** approach that preserve such structures under
# certain assumptions. The main idea is to filter a similarity matrix along various
# directions that lie in a neighborhood of the direction defined by the main diagonal.

# %%
@jit(nopython=True)
def compute_tempo_rel_set(tempo_rel_min, tempo_rel_max, num):
    """Compute logarithmically spaced relative tempo values

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        tempo_rel_min (float): Minimum relative tempo
        tempo_rel_max (float): Maximum relative tempo
        num (int): Number of relative tempo values (inlcuding the min and max)

    Returns:
        tempo_rel_set (np.ndarray): Set of relative tempo values
    """
    tempo_rel_set = np.exp(np.linspace(np.log(tempo_rel_min), np.log(tempo_rel_max), num))
    return tempo_rel_set

tempo_rel_min = 0.66
tempo_rel_max = 1.5
num = 5
tempo_rel_set = compute_tempo_rel_set(tempo_rel_min=tempo_rel_min, tempo_rel_max=tempo_rel_max, num=num)
print(tempo_rel_set)

# %%
@jit(nopython=True)
def filter_diag_mult_sm(S, L=1, tempo_rel_set=np.asarray([1]), direction=0):
    """Path smoothing of similarity matrix by filtering in forward or backward direction
    along various directions around main diagonal.
    Note: Directions are simulated by resampling one axis using relative tempo values

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        S (np.ndarray): Self-similarity matrix (SSM)
        L (int): Length of filter (Default value = 1)
        tempo_rel_set (np.ndarray): Set of relative tempo values (Default value = np.asarray([1]))
        direction (int): Direction of smoothing (0: forward; 1: backward) (Default value = 0)

    Returns:
        S_L_final (np.ndarray): Smoothed SM
    """
    N = S.shape[0]
    M = S.shape[1]
    num = len(tempo_rel_set)
    S_L_final = np.zeros((N, M))

    for s in range(0, num):
        M_ceil = int(np.ceil(M / tempo_rel_set[s]))
        resample = np.multiply(np.divide(np.arange(1, M_ceil+1), M_ceil), M)
        np.around(resample, 0, resample)
        resample = resample - 1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)
        S_resample = S[:, index_resample]

        S_L = np.zeros((N, M_ceil))
        S_extend_L = np.zeros((N + L, M_ceil + L))

        # Forward direction
        if direction == 0:
            S_extend_L[0:N, 0:M_ceil] = S_resample
            for pos in range(0, L):
                S_L = S_L + S_extend_L[pos:(N + pos), pos:(M_ceil + pos)]

        # Backward direction
        if direction == 1:
            S_extend_L[L:(N+L), L:(M_ceil+L)] = S_resample
            for pos in range(0, L):
                S_L = S_L + S_extend_L[(L-pos):(N + L - pos), (L-pos):(M_ceil + L - pos)]

        S_L = S_L / L
        resample = np.multiply(np.divide(np.arange(1, M+1), M), M_ceil)
        np.around(resample, 0, resample)
        resample = resample - 1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)

        S_resample_inv = S_L[:, index_resample]
        S_L_final = np.maximum(S_L_final, S_resample_inv)

    return S_L_final

tempo_rel_min = 0.66
tempo_rel_max = 1.5
num = 5
tempo_rel_set = compute_tempo_rel_set(tempo_rel_min=tempo_rel_min, tempo_rel_max=tempo_rel_max, num=num)
L = 20
S_L = filter_diag_sm(S, L)
S_L_mult = filter_diag_mult_sm(S, L, tempo_rel_set)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
subplot_matrix_colorbar(S, fig, ax[0], clim=[0,1], xlabel='Time (frames)', ylabel='Time (frames)',
                        title=r'Original SSM $\mathbf{S}$')
subplot_matrix_colorbar(S_L, fig, ax[1], clim=[0,1], xlabel='Time (frames)', ylabel='Time (frames)',
                        title=r'Smoothed SSM $\mathbf{S}_{L}$ (L = %d)'%L)
subplot_matrix_colorbar(S_L_mult, fig, ax[2], clim=[0,1], xlabel='Time (frames)', ylabel='Time (frames)',
                        title=r'Multiple filtering SSM $\mathbf{S}_{L,\theta}$ (L = %d)'%L)

plt.tight_layout()
plt.show()

# %% [markdown]
# In the following figure, we zoom into the SSMs comparing the B1-section (horizontal
# axis) with the B2-section (vertical axis).

# %%
xlim = [75, 150]
ylim = [125, 200]

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
subplot_matrix_colorbar(S, fig, ax[0], clim=[0,1], xlim=xlim, ylim=ylim,
                        title='Original SSM', xlabel='Time (frames)', ylabel='Time (frames)')
subplot_matrix_colorbar(S_L, fig, ax[1], clim=[0,1], xlim=xlim, ylim=ylim,
                        title='Smoothed SSM (L = %d)'%L, xlabel='Time (frames)', ylabel='Time (frames)')
subplot_matrix_colorbar(S_L_mult, fig, ax[2], clim=[0,1], xlim=xlim, ylim=ylim,
                        title=r'Multiple filtering SSM $\mathbf{S}_{L,\theta}$ (L = %d)'%L,
                        xlabel='Time (frames)', ylabel='Time (frames)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Forward-Backward Smoothing
#
# There is a fading artifact in particular at the end of path structures. The reason
# is that our smoothing procedure described so far works in the forward direction. To
# avoid fading artifacts, one idea is to additionally apply the averaging filter in a
# backward direction. The final self-similarity matrix is then obtained by taking the
# cell-wise maximum over the **forward-smoothed** and **backward-smoothed** matrices.

# %%
tempo_rel_min = 0.66
tempo_rel_max = 1.5
num = 5
tempo_rel_set = compute_tempo_rel_set(tempo_rel_min=tempo_rel_min, tempo_rel_max=tempo_rel_max, num=num)
L = 20
S_forward = filter_diag_mult_sm(S, L, tempo_rel_set, direction=0)
S_backward = filter_diag_mult_sm(S, L, tempo_rel_set, direction=1)
S_final = np.maximum(S_forward, S_backward)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
subplot_matrix_colorbar(S, fig, ax[0], clim=[0,1],
                        title=r'Original SSM', xlabel='Time (frames)', ylabel='Time (frames)')
subplot_matrix_colorbar(S_forward, fig, ax[1], clim=[0,1],
                        title=r'Forward SSM', xlabel='Time (frames)', ylabel='Time (frames)')
subplot_matrix_colorbar(S_final, fig, ax[2], clim=[0,1],
                        title=r'Forward-backward SSM', xlabel='Time (frames)', ylabel='Time (frames)')

plt.tight_layout()
plt.show()

# %% [markdown]
# Zoomed view comparing B1-section with B2-section:

# %%
xlim = [75, 150]
ylim = [125, 200]

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
subplot_matrix_colorbar(S, fig, ax[0], clim=[0,1], xlim=xlim, ylim=ylim,
                        title='Original SSM', xlabel='Time (frames)', ylabel='Time (frames)')
subplot_matrix_colorbar(S_forward, fig, ax[1], clim=[0,1], xlim=xlim, ylim=ylim,
                        title='Forward SSM', xlabel='Time (frames)', ylabel='Time (frames)')
subplot_matrix_colorbar(S_final, fig, ax[2], clim=[0,1], xlim=xlim, ylim=ylim,
                        title='Forward-backward SSM', xlabel='Time (frames)', ylabel='Time (frames)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Further Notes
#
# Most enhancement procedures are based on smoothing the given matrix along the direction
# of the main diagonal. Such a filtering process is closely related to the concept of
# **time-delay embedding**. Also **morphological operations** used in image processing
# can be applied for augmenting path-like structures. An alternative to our
# **resample-based approach** is to apply multiple **2D convolution kernels** as
# implemented in LibROSA.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and David Kopyto.

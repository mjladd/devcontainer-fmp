# %% [markdown]
# # Novelty-Based Segmentation
#
# Following Section 4.4.1 of [Müller, FMP, Springer 2015], we introduce in this notebook
# a basic procedure for novelty detection.

# %% [markdown]
# ## Introduction
#
# Segment boundaries in music are often accompanied by a change in instrumentation,
# dynamics, harmony, tempo, or some other characteristics. It is the objective of
# novelty-based structure analysis to locate points in time where such musical changes
# occur.

# %%
import numpy as np
import os, sys, librosa
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib
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

# SM
fn_wav = os.path.join('..', 'data', 'C4', 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.wav')
tempo_rel_set = libfmp.c4.compute_tempo_rel_set(0.66, 1.5, 5)
x, x_duration, X, Fs_X, S, I = libfmp.c4.compute_sm_from_filename(fn_wav,
                                                L=81, H=10, L_smooth=1, thresh=1)

# Visualization
ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X)
fig, ax = libfmp.c4.plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_X,
            label='Time (frames)', color_ann=color_ann, clim_X=[0,1], clim=[0,1],
            title='Feature rate: %0.0f Hz'%(Fs_X))

# %% [markdown]
# ## Overall Procedure
#
# The idea of Foote's procedure is to measure local changes by correlating a small
# checkerboard-like kernel along the main diagonal of an SSM. This results in a
# **novelty function** that reveals a peak at time positions where the kernel meets
# a transition between two contrasting blocks.

# %% [markdown]
# ## Checkerboard Kernel: Box
#
# To find the boundary between two sections one needs to identify the crux of the
# checkerboard. This can be done by correlating the SSM with a kernel that itself
# looks like a checkerboard.

# %%
def compute_kernel_checkerboard_box(L):
    """Compute box-like checkerboard kernel [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L (int): Parameter specifying the kernel size 2*L+1

    Returns:
        kernel (np.ndarray): Kernel matrix of size (2*L+1) x (2*L+1)
    """
    axis = np.arange(-L, L+1)
    kernel = np.outer(np.sign(axis), np.sign(axis))
    return kernel

L = 10
kernel = compute_kernel_checkerboard_box(L)
plt.figure(figsize=(4,3))
plt.imshow(kernel, aspect='auto', origin='lower',
           extent=[-L-0.5,L+0.5,-L-0.5,L+0.5], cmap='seismic')
plt.colorbar()
plt.tight_layout()

# %% [markdown]
# ## Checkerboard Kernel: Gaussian
#
# The checkerboard kernel can be smoothed to avoid edge effects using windows that
# taper towards zero at the edges.

# %%
@jit(nopython=True)
def compute_kernel_checkerboard_gaussian(L, var=1, normalize=True):
    """Compute Gaussian-like checkerboard kernel [FMP, Section 4.4.1].

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L (int): Parameter specifying the kernel size M=2*L+1
        var (float): Variance parameter determining the tapering (epsilon) (Default value = 1.0)
        normalize (bool): Normalize kernel (Default value = True)

    Returns:
        kernel (np.ndarray): Kernel matrix of size M x M
    """
    taper = np.sqrt(1/2) / (L * var)
    axis = np.arange(-L, L+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel

L = 10
var = 0.5
kernel = compute_kernel_checkerboard_gaussian(L, var)
plt.figure(figsize=(4,3))
plt.imshow(kernel, aspect='auto', origin='lower',
           extent=[-L-0.5,L+0.5,-L-0.5,L+0.5], cmap='seismic')
plt.colorbar()
plt.tight_layout()

# %% [markdown]
# ## Novelty Function
#
# To detect 2D corner points between adjoining blocks, the idea is to locally compare
# the SSM with a checkerboard kernel.

# %%
def compute_novelty_ssm(S, kernel=None, L=10, var=0.5, exclude=False):
    """Compute novelty function from SSM [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        S (np.ndarray): SSM
        kernel (np.ndarray): Checkerboard kernel (if kernel==None, it will be computed) (Default value = None)
        L (int): Parameter specifying the kernel size M=2*L+1 (Default value = 10)
        var (float): Variance parameter determining the tapering (epsilon) (Default value = 0.5)
        exclude (bool): Sets the first L and last L values of novelty function to zero (Default value = False)

    Returns:
        nov (np.ndarray): Novelty function
    """
    if kernel is None:
        kernel = compute_kernel_checkerboard_gaussian(L=L, var=var)
    N = S.shape[0]
    M = 2*L + 1
    nov = np.zeros(N)
    S_padded = np.pad(S, L, mode='constant')

    for n in range(N):
        nov[n] = np.sum(S_padded[n:n+M, n:n+M] * kernel)
    if exclude:
        right = np.min([L, N])
        left = np.max([0, N-L])
        nov[0:right] = 0
        nov[left:N] = 0

    return nov

L_kernel = 20
nov = compute_novelty_ssm(S, L=L_kernel, exclude=False)
fig, ax, line = libfmp.b.plot_signal(nov, Fs = Fs_X, color='k')

# %%
L_kernel = 20
nov = compute_novelty_ssm(S, L=L_kernel, exclude=True)
fig, ax, line = libfmp.b.plot_signal(nov, Fs = Fs_X, color='k')
libfmp.b.plot_segments_overlay(ann, ax=ax, colors=color_ann, alpha=0.1, edgecolor='k', print_labels=False)
plt.tight_layout()

# %% [markdown]
# ## Kernel Size
#
# The size of the kernel has a significant impact on the properties of the novelty
# function. A **small kernel** may be suitable for detecting novelty on a short time
# scale, whereas a **large kernel** is suited for detecting boundaries and transitions
# between coarse structural sections.

# %%
from libfmp.b import FloatingBox
float_box = libfmp.b.FloatingBox()

fn_wav = os.path.join('..', 'data', 'C4', 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.wav')

S_dict = {}
Fs_dict = {}
x, x_duration, X, Fs_X, S, I = libfmp.c4.compute_sm_from_filename(fn_wav,
                                                L=11, H=5, L_smooth=1, thresh=1)
S_dict[0], Fs_dict[0] = S, Fs_X
ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X)
fig, ax = libfmp.c4.plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_X,
            label='Time (frames)', color_ann=color_ann, clim_X=[0,1], clim=[0,1],
            title='Feature rate: %0.0f Hz'%(Fs_X), figsize=(4.5, 5.5))
float_box.add_fig(fig)

x, x_duration, X, Fs_X, S, I = libfmp.c4.compute_sm_from_filename(fn_wav,
                                                L=41, H=10, L_smooth=1, thresh=1)
S_dict[1], Fs_dict[1] = S, Fs_X
ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X)
fig, ax = libfmp.c4.plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_X,
            label='Time (frames)', color_ann=color_ann, clim_X=[0,1], clim=[0,1],
            title='Feature rate: %0.0f Hz'%(Fs_X), figsize=(4.5, 5.5))
float_box.add_fig(fig)
float_box.show()


figsize=(10,6)
L_kernel_set = [5, 10, 20, 40]
num_kernel = len(L_kernel_set)
num_SSM = len(S_dict)

fig, ax = plt.subplots(num_kernel, num_SSM, figsize=figsize)
for s in range(num_SSM):
    for t in range(num_kernel):
        L_kernel = L_kernel_set[t]
        S = S_dict[s]
        nov = compute_novelty_ssm(S, L=L_kernel, exclude=True)
        fig_nov, ax_nov, line_nov = libfmp.b.plot_signal(nov, Fs = Fs_dict[s],
                color='k', ax=ax[t,s], figsize=figsize,
                title='Feature rate = %0.0f Hz, $L_\mathrm{kernel}$ = %d'%(Fs_dict[s],L_kernel))
        libfmp.b.plot_segments_overlay(ann, ax=ax_nov, colors=color_ann, alpha=0.1,
                                       edgecolor='k', print_labels=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Further Notes
#
# In the novelty detection procedure, there are many choices including the feature
# representation and the kernel size. Moreover, the **peak selection strategy** is
# also a delicate step that may have a substantial influence on the quality of the
# final result.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Julian Reck.

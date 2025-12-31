# %% [markdown]
# # Structure Feature
#
# Following Section 4.4.2 of [Müller, FMP, Springer 2015], we discuss in this notebook
# the concept of structure features.

# %% [markdown]
# ## Basic Idea
#
# The idea behind structure features is to jointly consider local and global aspects
# by measuring for each frame of a given feature sequence the relations to all other
# frames of the same feature sequence.

# %% [markdown]
# ## Time-Lag Representation
#
# We start by introducing the concept of **time-lag matrices**, which is the main
# technical ingredient for defining the structure features.

# %%
import numpy as np
import os, sys, librosa
from scipy import signal
from scipy import ndimage
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

def compute_time_lag_representation(S, circular=True):
    """Computation of (circular) time-lag representation

    Notebook: C4/C4S4_StructureFeature.ipynb

    Args:
        S (np.ndarray): Self-similarity matrix
        circular (bool): Computes circular version (Default value = True)

    Returns:
        L (np.ndarray): (Circular) time-lag representation of S
    """
    N = S.shape[0]
    if circular:
        L = np.zeros((N, N))
        for n in range(N):
            L[:, n] = np.roll(S[:, n], -n)
    else:
        L = np.zeros((2*N-1, N))
        for n in range(N):
            L[((N-1)-n):((2*N)-1-n), n] = S[:, n]
    return L

ann = [[0, 9, 'A'], [10, 19, 'B'], [20, 29, 'A'], [30, 39, 'B'], [40, 49, 'B'], [50, 59, 'A']]
S = libfmp.c4.generate_ssm_from_annotation(ann, score_path=1, score_block=0.3)
N = S.shape[0]
L = compute_time_lag_representation(S, circular=False)
L_circ = compute_time_lag_representation(S, circular=True)

plt.figure(figsize=(5, 5))

ax1 = plt.axes([0, 0.6, 0.4, 0.4])
ax2 = plt.axes([0, 0, 0.4, 0.4])
ax3 = plt.axes([0.6, 0.0, 0.4, 0.8])

fig, ax, im = libfmp.b.plot_matrix(S, ax=[ax1], title='SSM',
                                   xlabel='Time (frames)', ylabel='Time (frames)', colorbar=None)

fig, ax, im = libfmp.b.plot_matrix(L_circ, ax=[ax2], title='Circular time-lag matrix',
                                   xlabel='Time (frames)', ylabel='Lag (frames)', colorbar=None)

fig, ax, im = libfmp.b.plot_matrix(L, ax=[ax3], extent=[0, N-1, -(N-1), N-1], title='Time-lag matrix',
                                   xlabel='Time (frames)', ylabel='Lag (frames)', colorbar=None)

# %%
ann = [[0, 9, 'A'], [10, 19, 'B'], [20, 39, 'B']]
S = libfmp.c4.generate_ssm_from_annotation(ann, score_path=1, score_block=0.3)
N = S.shape[0]
L = compute_time_lag_representation(S, circular=False)
L_circ = compute_time_lag_representation(S, circular=True)

plt.figure(figsize=(5, 2.5))
ax = plt.subplot(121)
fig, ax, im = libfmp.b.plot_matrix(S, ax=[ax], title='SSM',
                                   xlabel='Time (frames)', ylabel='Time (frames)', colorbar=None)
ax = plt.subplot(122)
fig, ax, im = libfmp.b.plot_matrix(L_circ, ax=[ax], title='Circular time-lag matrix',
                                   xlabel='Time (frames)', ylabel='Lag (frames)', colorbar=None)
plt.tight_layout()

# %% [markdown]
# ## Structure Features and Novelty Function
#
# We define the **structure features** to be the columns of the circular time-lag
# representation. By computing the difference between successive structure features,
# one obtains a **novelty function**.

# %%
def novelty_structure_feature(L, padding=True):
    """Computation of the novelty function from a circular time-lag representation

    Notebook: C4/C4S4_StructureFeature.ipynb

    Args:
        L (np.ndarray): Circular time-lag representation
        padding (bool): Padding the result with the value zero (Default value = True)

    Returns:
        nov (np.ndarray): Novelty function
    """
    N = L.shape[0]
    if padding:
        nov = np.zeros(N)
    else:
        nov = np.zeros(N-1)
    for n in range(N-1):
        nov[n] = np.linalg.norm(L[:, n+1] - L[:, n])
    return nov

def plot_ssm_structure_feature_nov(S, L, nov, Fs=1, figsize=(10, 3), ann=[], color_ann=[]):
    """Plotting an SSM, structure features, and a novelty function

    Notebook: C4/C4S4_StructureFeature.ipynb
    """
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(131)
    if Fs == 1:
        title = 'SSM'
    else:
        title = 'SSM (Fs = %d)' % Fs
    fig_im, ax_im, im = libfmp.b.plot_matrix(S, ax=[ax1], title=title,
                                             xlabel='Time (frames)', ylabel='Time (frames)')
    if ann:
        libfmp.b.plot_segments_overlay(ann, ax=ax_im[0], edgecolor='k',
                                       print_labels=False, colors=color_ann, alpha=0.05)

    ax2 = plt.subplot(132)
    fig_im, ax_im, im = libfmp.b.plot_matrix(L, ax=[ax2], title='Structure features',
                                             xlabel='Time (frames)', ylabel='Lag (frames)', colorbar=True)
    if ann:
        libfmp.b.plot_segments_overlay(ann, ax=ax_im[0], edgecolor='k', ylim=False,
                                       print_labels=False, colors=color_ann, alpha=0.05)

    ax3 = plt.subplot(133)
    fig, ax, im = libfmp.b.plot_signal(nov, ax=ax3, title='Novelty function',
                                       xlabel='Time (frames)', color='k')
    if ann:
        libfmp.b.plot_segments_overlay(ann, ax=ax, edgecolor='k', colors=color_ann, alpha=0.05)
    plt.tight_layout()
    return ax1, ax2, ax3

color_ann = {'A': [1, 0, 0, 0.2], 'B': [0, 1, 0, 0.2], 'C': [0, 0, 1, 0.2], '': [1, 1, 1, 0.2]}
ann = [[0, 9, 'A'], [10, 19, 'B'], [20, 29, 'A'], [30, 39, 'B'], [40, 49, 'B'], [50, 59, 'A']]
S = libfmp.c4.generate_ssm_from_annotation(ann, score_path=1, score_block=0)
L = compute_time_lag_representation(S, circular=True)
nov = novelty_structure_feature(L)
ax = plot_ssm_structure_feature_nov(S, L, nov, ann=ann, color_ann=color_ann)

# %%
color_ann = {'A': [1, 0, 0, 0.2], 'B': [0, 1, 0, 0.2], 'C': [0, 0, 1, 0.2], '': [1, 1, 1, 0.2]}
figsize = (8,2.5)

ann_set = [[[0, 9, 'A'], [10, 19, 'A'], [20, 29, 'A'], [30, 39, 'A']],
       [[0, 9, 'A'], [10, 19, 'B'], [20, 29, 'A'], [30, 39, 'B']],
       [[0, 9, 'A'], [10, 19, 'B'], [20, 29, 'A'], [30, 39, 'A']],
       [[0, 9, 'A'], [10, 19, 'A'], [20, 29, 'B'], [30, 39, 'B']]]

for ann in ann_set:
    S = libfmp.c4.generate_ssm_from_annotation(ann, score_path=1, score_block=0)
    L = compute_time_lag_representation(S, circular=True)
    nov = novelty_structure_feature(L)
    ax = plot_ssm_structure_feature_nov(S, L, nov, figsize=figsize, ann=ann, color_ann=color_ann)

# %% [markdown]
# ## Example: Chopin

# %%
fn_wav = os.path.join('..', 'data', 'C4', 'FMP_C4_Audio_Chopin_Op028-11_003_20100611-SMD.wav')
fn_ann = os.path.join('..', 'data', 'C4', 'FMP_C4_Audio_Chopin_Op028-11_003_20100611-SMD.csv')
x, x_duration, X, Fs_feature, S, I = libfmp.c4.compute_sm_from_filename(fn_wav, L=9, H=2, L_smooth=11, thresh= 0.1)

ann_frames, color_ann = libfmp.c4.read_structure_annotation(fn_ann, Fs=Fs_feature, remove_digits=True, index=True)
color_ann = {'A': [1, 0, 0, 0.2], 'B': [0, 1, 0, 0.2], 'C': [0, 0, 1, 0.2], '': [1, 1, 1, 0.2]}

L = compute_time_lag_representation(S, circular=True)
nov = novelty_structure_feature(L)
ax = plot_ssm_structure_feature_nov(S, L, nov, Fs=Fs_feature, ann=ann_frames, color_ann=color_ann)

# %%
print('Application of median filter')
L_filter = ndimage.median_filter(L, (3,11))
nov = novelty_structure_feature(L_filter)
ax = plot_ssm_structure_feature_nov(S, L_filter, nov, ann=ann_frames, color_ann=color_ann)
plt.show()

print('Application of convolution filter')
L_filter = ndimage.gaussian_filter(L_filter, 4)
nov = novelty_structure_feature(L_filter)
ax = plot_ssm_structure_feature_nov(S, L_filter, nov, ann=ann_frames, color_ann=color_ann)

# %% [markdown]
# ## Example: Brahms

# %%
filename = 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.csv'
fn_ann = os.path.join('..', 'data', 'C4', filename)
fn_wav = os.path.join('..', 'data', 'C4', 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.wav')
tempo_rel_set = np.array([0.8, 1, 1.25])
x, x_duration, X, Fs_feature, S, I = libfmp.c4.compute_sm_from_filename(fn_wav, L=21, H=5,
                                    L_smooth=11, tempo_rel_set=tempo_rel_set, thresh= 0.1)

ann_frames, color_ann = libfmp.c4.read_structure_annotation(fn_ann, fn_ann_color=filename,
                                                            Fs=Fs_feature, remove_digits=True, index=True)

L = compute_time_lag_representation(S, circular=True)
nov = novelty_structure_feature(L)
ax = plot_ssm_structure_feature_nov(S, L, nov, Fs=Fs_feature, ann=ann_frames, color_ann=color_ann)

# %%
L_filter = ndimage.median_filter(L, (3,21))
L_filter = ndimage.gaussian_filter(L_filter, 6)
nov = novelty_structure_feature(L_filter)
ax = plot_ssm_structure_feature_nov(S, L_filter, nov, Fs=Fs_feature, ann=ann_frames, color_ann=color_ann)

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Tim Zunner.

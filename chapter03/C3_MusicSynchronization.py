# %% [markdown]
# # Music Synchronization
#
# In this notebook, we give an example for music synchronization following Chapter 3
# of [Müller, FMP, Springer 2015]. For technical details we refer to Section 3.1
# (Audio Features) and Section 3.2 (Dynamic Time Warping).

# %% [markdown]
# ## Introduction
#
# Music can be described and represented in many different ways including sheet music,
# symbolic representations, and audio recordings. For each of these representations,
# there may exist different versions that correspond to the same musical work.
#
# Given two different music representations, typical synchronization approaches
# proceed in two steps:
#
# - In the first step, the two representations are transformed into sequences of
#   suitable features. We use **chroma-based features**, which capture harmonic and
#   melodic characteristics of music, while being robust to changes in timbre and
#   instrumentation.
#
# - In the second step, the derived feature sequences have to be brought into temporal
#   correspondence using **dynamic time warping** (DTW).

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
%matplotlib inline

import sys
sys.path.append('..')
import libfmp.c3

Fs = 22050
fn_wav_X = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-21_Midi-Piano.wav')
fn_wav_Y = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-21_Karajan1946.wav')
X_wav, Fs = librosa.load(fn_wav_X, sr=Fs)
Y_wav, Fs = librosa.load(fn_wav_Y, sr=Fs)

N = 4410
H = 2205
X = librosa.feature.chroma_stft(y=X_wav, sr=Fs, tuning=0, norm=2, hop_length=H, n_fft=N)
Y = librosa.feature.chroma_stft(y=Y_wav, sr=Fs, tuning=0, norm=2, hop_length=H, n_fft=N)

plt.figure(figsize=(8, 2))
plt.title('Sequence $X$')
librosa.display.specshow(X, x_axis='frames', y_axis='chroma', cmap='gray_r', hop_length=H)
plt.xlabel('Time (frames)')
plt.ylabel('Chroma')
plt.colorbar()
plt.clim([0, 1])
plt.tight_layout()
plt.show()
ipd.display(ipd.Audio(X_wav, rate=Fs))

plt.figure(figsize=(8, 2))
plt.title('Sequence $Y$')
librosa.display.specshow(Y, x_axis='frames', y_axis='chroma', cmap='gray_r', hop_length=H)
plt.xlabel('Time (frames)')
plt.ylabel('Chroma')
plt.colorbar()
plt.clim([0, 1])
plt.tight_layout()
plt.show()
ipd.display(ipd.Audio(Y_wav, rate=Fs))

# %% [markdown]
# ## Application of DTW
#
# Next, we compute the cost matrix C based on the Euclidean distance, then the
# accumulated cost matrix D using dynamic programming, and finally an optimal
# warping path P* using backtracking.

# %%
C = libfmp.c3.compute_cost_matrix(X, Y)
D = libfmp.c3.compute_accumulated_cost_matrix(C)
P = libfmp.c3.compute_optimal_warping_path(D)

plt.figure(figsize=(12, 3))
ax = plt.subplot(1, 2, 1)
libfmp.c3.plot_matrix_with_points(C, P, linestyle='-', marker='',
    ax=[ax], aspect='equal', clim=[0, np.max(C)],
    title='$C$ with optimal warping path', xlabel='Sequence Y', ylabel='Sequence X')

ax = plt.subplot(1, 2, 2)
libfmp.c3.plot_matrix_with_points(D, P, linestyle='-', marker='',
    ax=[ax], aspect='equal', clim=[0, np.max(D)],
    title='$D$ with optimal warping path', xlabel='Sequence Y', ylabel='Sequence X')

plt.tight_layout()

# %% [markdown]
# Finally, we give some alternative for visualizing the final result of the alignment.

# %%
N = X.shape[1]
M = Y.shape[1]

plt.figure(figsize=(8, 3))
ax_X = plt.axes([0, 0.60, 1, 0.40])
librosa.display.specshow(X, ax=ax_X, x_axis='frames', y_axis='chroma', cmap='gray_r', hop_length=H)
ax_X.set_ylabel('Sequence X')
ax_X.set_xlabel('Time (frames)')
ax_X.xaxis.tick_top()
ax_X.xaxis.set_label_position('top')

ax_Y = plt.axes([0, 0, 1, 0.40])
librosa.display.specshow(Y, ax=ax_Y, x_axis='frames', y_axis='chroma', cmap='gray_r', hop_length=H)
ax_Y.set_ylabel('Sequence Y')
ax_Y.set_xlabel('Time (frames)')

step = 5
y_min_X, y_max_X = ax_X.get_ylim()
y_min_Y, y_max_Y = ax_Y.get_ylim()
for t in P[0:-1:step, :]:
    ax_X.vlines(t[0], y_min_X, y_max_X, color='r')
    ax_Y.vlines(t[1], y_min_Y, y_max_Y, color='r')

ax = plt.axes([0, 0.40, 1, 0.20])
for p in P[0:-1:step, :]:
    ax.plot((p[0]/N, p[1]/M), (1, -1), color='r')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
ax.set_xticks([])
ax.set_yticks([])

# %% [markdown]
# ## Further Notes
#
# In this notebook, we introduced a basic approach for synchronizing different
# recordings of the same piece of music. Music synchronization and related alignment
# tasks have been studied extensively within the field of music information retrieval.
# Depending upon the respective types of music representations, one can distinguish
# between various synchronization scenarios including:
#
# - **Audio-audio synchronization**: Alignment of two music recordings
# - **Score-audio synchronization**: Alignment of symbolically encoded score events
#   with time positions in a recording
# - **Image-audio synchronization**: Alignment of spatial positions of digitized
#   images of sheet music with time positions of a recording
# - **Lyrics-audio synchronization**: Alignment of lyrics with time positions of
#   a recorded song

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Stefan Balke.

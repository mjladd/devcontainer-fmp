# %% [markdown]
# # Application: Tempo Curves
#
# Following Section 3.3.2 of [Müller, FMP, Springer 2015], we discuss in this notebook
# how synchronization results can be used for automatically computing the local tempo
# over time of an expressive music recording.

# %% [markdown]
# ## Performance Analysis
#
# Musicians give a piece of music their personal touch by continuously varying tempo,
# dynamics, and articulation. Instead of playing mechanically, they speed up at some
# places and slow down at others in order to shape a piece of music.

# %% [markdown]
# ## Tempo Curve Computation: Basic Idea
#
# We now present an approach for deriving tempo-related information using music
# synchronization techniques. The overall procedure works as follows:
#
# * First, the **reference version** is converted into a piano-roll presentation, where
#   the notes are played with a known constant tempo in a purely mechanical way.
# * Then, music synchronization techniques are used to temporally align the note and beat
#   events of the reference with their corresponding physical occurrences in a given
#   performed audio version.
# * The resulting warping path reveals the **relative tempo differences** between the
#   actual performance and the neutral reference version.

# %%
import os
import sys
import copy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import librosa
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import scipy.ndimage.filters
%matplotlib inline

sys.path.append('..')
import libfmp.b
import libfmp.c1
import libfmp.c3

P = np.array([[0, 0], [2, 1], [3, 2], [3.4, 3], [3.7, 4], [4, 5]])
bpm = []

pos_audio1, pos_score1 = P[0, :]
for pos_audio2, pos_score2  in P[1:]:
    dur_audio = pos_audio2 - pos_audio1
    dur_score = pos_score2 - pos_score1
    bpm.append(60 * (dur_score / dur_audio))
    pos_audio1, pos_score1 = pos_audio2, pos_score2

fig, ax = plt.subplots(1, 2, figsize=(8, 3))

ax[0].plot(P[:, 1], P[:, 0], 'r-', linewidth=2)
ax[0].set_yticks([0, 1, 2, 3.0, 3.4, 3.7, 4.0])
ax[0].grid()
ax[0].set_xlim([0, 5])
ax[0].set_ylim([0, 4])
ax[0].set_xlabel('Time (beats)')
ax[0].set_ylabel('Time (seconds)')
ax[0].set_title('Warping path')

ax[1].plot(np.arange(len(bpm)) + 0.5, bpm, 'k-')
ax[1].plot(np.arange(len(bpm)) + 0.5, bpm, 'ro', markersize=10)
ax[1].set_yticks([30, 60, 150, 200])
ax[1].grid()
ax[1].set_xlim([0, 5])
ax[1].set_ylim([0, 240])
ax[1].set_xlabel('Time (beats)')
ax[1].set_ylabel('Tempo (BPM)')
ax[1].set_title('Tempo curve')
plt.tight_layout()

# %% [markdown]
# ## Example: Schumann's Träumerei
#
# As an example, we consider three performances of the first eight measures of the
# "Träumerei" by Robert Schumann.

# %% [markdown]
# ## Computation: Chromagram
#
# Following the approach as described, we now describe a baseline implementation for
# computing tempo curve.

# %%
def compute_score_chromagram(score, Fs_beat):
    """Compute chromagram from score representation

    Notebook: C3/C3S3_MusicAppTempoCurve.ipynb

    Args:
        score (list): Score representation
        Fs_beat (scalar): Sampling rate for beat axis

    Returns:
        X_score (np.ndarray): Chromagram representation X_score
        t_beat (np.ndarray): Time axis t_beat (given in beats)
    """
    score_beat_min = min(n[0] for n in score)
    score_beat_max = max(n[0] + n[1] for n in score)
    beat_res = 1.0 / Fs_beat
    t_beat = np.arange(score_beat_min, score_beat_max, beat_res)
    X_score = np.zeros((12, len(t_beat)))

    for start, duration, pitch, velocity, label in score:
        start_idx = int(round(start / beat_res))
        end_idx = int(round((start + duration) / beat_res))
        cur_chroma = int(round(pitch)) % 12
        X_score[cur_chroma, start_idx:end_idx] += velocity

    X_score = librosa.util.normalize(X_score, norm=2)
    return X_score, t_beat

def plot_measure(ax, measure_pos):
    """Plot measure positions

    Notebook: C3/C3S3_MusicAppTempoCurve.ipynb

    Args:
        ax (mpl.axes.Axes): Figure axis
        measure_pos (list or np.ndarray): Array containing measure positions
    """
    y_min, y_max = ax.get_ylim()
    ax.vlines(measure_pos, y_min, y_max, color='r')
    for m in range(len(measure_pos)):
        ax.text(measure_pos[m], y_max, '%s' % (m + 1),
                color='r', backgroundcolor='mistyrose',
                verticalalignment='top', horizontalalignment='left')

# Read score file and plot piano-roll representation
fn_score = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Schumann_Op15No7_Traeumerei_MusicXML.csv')
measure_pos_beat = np.array([1, 5, 9, 13, 17, 21, 25, 29, 33])
df = pd.read_csv(fn_score, sep=';')
print('First six note events of score CVS file:')
print(df.loc[0:5, :])

score = libfmp.c1.csv_to_list(fn_score)
fig, ax = libfmp.c1.visualize_piano_roll(score, xlabel='Time (beats)',
                                         ylabel='Pitch (note numbers)',
                                         colors=['gray'],
                                         velocity_alpha=False, figsize=(6.6, 3))
ax.set_title('Piano-roll representation for score')
ax.get_legend().remove()
plot_measure(ax, measure_pos_beat)
plt.tight_layout()

 # Compute and plot score-based chromagram
Fs_beat = 10
X_score, t_beat = compute_score_chromagram(score, Fs_beat)
figsize = (8, 2)
fig, ax, im = libfmp.b.plot_chromagram(X_score, Fs=1, figsize=figsize,
                                       xlabel='Time (frames)',
                                       title='Chromagram for score ($F_\\mathrm{s} = %.1f$)'%Fs_beat,
                                       chroma_yticks=[0, 4, 7, 11])
plot_measure(ax[0], measure_pos_beat * Fs_beat)
plt.tight_layout()

# Read signal
Fs = 22050
fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Schumann_Op15No7_Traeumerei_HernandezRomero.wav')
x, Fs = librosa.load(fn_wav)

# Compute and plot chromagram
N = 4410
H = 2205
Fs_X = Fs / 2205
X = librosa.feature.chroma_stft(y=x, sr=Fs, norm=2, tuning=0, hop_length=H, n_fft=N)
libfmp.b.plot_chromagram(X, figsize=figsize, xlabel='Time (frames)', clim=[0, 1],
                         title='Chromagram for audio recording ($F_\\mathrm{s} = %.1f$)'%Fs_X,
                         chroma_yticks=[0, 4, 7, 11])
plt.tight_layout()

# %% [markdown]
# ## Computation: Alignment Path
#
# Based on the chromagrams, we apply dynamic time warping (DTW) for synchronizing the
# audio version to the score-based reference version.

# %%
# Compute DTW and plot C, D, P
C = libfmp.c3.compute_cost_matrix(X, X_score, metric='euclidean')
sigma = np.array([[1, 0], [0, 1], [1, 1]])
D, P = librosa.sequence.dtw(C=C, step_sizes_sigma=sigma)
P = P[::-1, :]  # reverse P

fig, ax = plt.subplots(1, 3, figsize=(11, 3))
libfmp.c3.plot_matrix_with_points(C, P, linestyle=':', marker='.', ax=[ax[0]],
                                  ylabel='Performance (audio)', xlabel='Reference (score)',
                                  title='$C$ and $P$', aspect='equal')
libfmp.c3.plot_matrix_with_points(D, P, linestyle=':', marker='.', ax=[ax[1]],
                                  ylabel='Performance (audio)', xlabel='Reference (score)',
                                  title='$D$ and $P$', aspect='equal')
libfmp.c3.plot_matrix_with_points(C, P, linestyle=':', marker='.', ax=[ax[2]],
                                  ylabel='Performance (audio)', xlabel='Reference (score)',
                                  title='$C$ and $P$ (zoom)', aspect='equal')
zoom_beat = np.array([4.8, 6.2])
zoom_sec = np.array([4.9, 6.2])
ax[2].set_xlim(zoom_beat * Fs_beat)
ax[2].set_ylim(zoom_sec * Fs_X)
plt.tight_layout()

# %% [markdown]
# To compute relative tempo differences, one basically needs to derive the slope of
# the warping path and then take its reciprocal to derive the tempo. We remove
# degenerations by deleting all horizontal and vertical sections. As a result, we
# obtain a modified path that is **strictly monotonous** in both dimension.

# %%
def compute_strict_alignment_path(P):
    """Compute strict alignment path from a warping path

    Notebook: C3/C3S3_MusicAppTempoCurve.ipynb

    Args:
        P (list or np.ndarray): Warping path

    Returns:
        P_mod (list or np.ndarray): Strict alignment path
    """
    # Initialize P_mod and enforce start boundary condition
    P_mod = np.zeros(P.shape)
    P_mod[0] = P[0]
    N, M = P[-1]
    # Go through all cells of P until reaching last row or column
    assert N > 1 and M > 1, 'Length of sequences must be longer than one.'
    i, j = 0, 0
    n1, m1 = P[i]
    while True:
        i += 1
        n2, m2 = P[i]
        if n2 == N or m2 == M:
            # If last row or column is reached, quit loop
            break
        if n2 > n1 and m2 > m1:
            # Strict monotonicity condition is fulfuilled
            j += 1
            P_mod[j] = n2, m2
            n1, m1 = n2, m2
    j += 1
    # Enforce end boundary condition
    P_mod[j] = P[-1]
    P_mod = P_mod[:j+1]
    return P_mod

def compute_strict_alignment_path_mask(P):
    """Compute strict alignment path from a warping path

    Notebook: C3/C3S3_MusicAppTempoCurve.ipynb

    Args:
        P (list or np.ndarray): Wapring path

    Returns:
        P_mod (list or np.ndarray): Strict alignment path
    """
    P = np.array(P, copy=True)
    N, M = P[-1]
    # Get indices for strict monotonicity
    keep_mask = (P[1:, 0] > P[:-1, 0]) & (P[1:, 1] > P[:-1, 1])
    # Add first index to enforce start boundary condition
    keep_mask = np.concatenate(([True], keep_mask))
    # Remove all indices for of last row or column
    keep_mask[(P[:, 0] == N) | (P[:, 1] == M)] = False
    # Add last index to enforce end boundary condition
    keep_mask[-1] = True
    P_mod = P[keep_mask, :]

    return P_mod

# Small toy example for testing
print('Toy example:')
X_toy =  [1, 3, 9, 2, 1]
Y_toy = [2, 0, 0, 8, 7, 2]
C_toy = libfmp.c3.compute_cost_matrix(X_toy, Y_toy, metric='euclidean')
sigma = np.array([[1, 0], [0, 1], [1, 1]])
D_toy, P_toy = librosa.sequence.dtw(C=C_toy, step_sizes_sigma=sigma)
P_toy = P_toy[::-1, :]  # reverse P
P_toy_mod =  compute_strict_alignment_path(P_toy)

fig, ax = plt.subplots(1, 2, figsize=(8, 2.5))
libfmp.c3.plot_matrix_with_points(C_toy, P_toy, linestyle=':', marker='.', ax=[ax[0]],
                                  ylabel='X', xlabel='Y',
                                  title='$C$ and $P$', aspect='equal')

libfmp.c3.plot_matrix_with_points(C_toy, P_toy_mod, linestyle=':', marker='.', ax=[ax[1]],
                                  ylabel='X', xlabel='Y',
                                  title='$C$ and strict alignment path', aspect='equal')
plt.tight_layout()
plt.show()

# Continuation with Schumann example
print('Schumann example (zoom):')
P_mod =  compute_strict_alignment_path(P)

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
libfmp.c3.plot_matrix_with_points(C, P, linestyle=':', marker='.', ax=[ax[0]],
                                  ylabel='Performance (audio)', xlabel='Reference (score)',
                                  title='$C$ and $P$ (zoom)', aspect='equal')
ax[0].set_xlim(zoom_beat * Fs_beat)
ax[0].set_ylim(zoom_sec * Fs_X)

libfmp.c3.plot_matrix_with_points(C, P_mod, linestyle=':', marker='.', ax=[ax[1]],
                                  ylabel='Performance (audio)', xlabel='Reference (score)',
                                  title='$C$ and strict alignment path (zoom)', aspect='equal')
ax[1].set_xlim(zoom_beat * Fs_beat)
ax[1].set_ylim(zoom_sec * Fs_X)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Computation: Tempo Curve
#
# The strict alignment path can be regarded as a strictly increasing function from the
# reference axis (given in beats) to the time axis of the performance (given in seconds).

# %%
# Convert path into beat-time function and interpolte
t_path_beat = P_mod[:, 1] / Fs_beat
f_path_sec = P_mod[:, 0] / Fs_X
f_sec = interp1d(t_path_beat, f_path_sec, kind='linear', fill_value='extrapolate')(t_beat)

# Compute difference and smooth with Hann window
f_diff_sec = np.diff(f_sec) * Fs_beat
f_diff_sec = np.concatenate((f_diff_sec, np.array([0])))
win_len_beat = 1
filt_len = int(win_len_beat * Fs_beat * 1.1)
filt_win = signal.windows.hann(filt_len)
filt_win = filt_win / np.sum(filt_win)
f_diff_smooth_sec = scipy.ndimage.filters.convolve(f_diff_sec, filt_win, mode='nearest')

fig, ax = plt.subplots(1, 3, figsize=(10, 2.5))
ax[0].plot(t_beat, f_sec, 'r')
ax[0].plot(t_path_beat, f_path_sec, color='r',  marker='o', linestyle='')
ax[0].set_title('Interpolated beat-time function')
ax[0].set_xlim(zoom_beat)
ax[0].set_ylim(zoom_sec)
ax[0].set_xlabel('Time (beats)')
ax[0].set_ylabel('Time (seconds)')

ax[1].plot(t_beat, f_diff_sec, 'r')
ax[1].set_title('Beat-duration function')
ax[1].set_xlim(zoom_beat)
ax[1].set_ylim([0.25, 2.1])
ax[1].set_xlabel('Time (beats)')
ax[1].set_ylabel('Time (seconds)')

ax[2].plot(t_beat, f_diff_smooth_sec, 'r')
ax[2].set_title('Smoothed beat-duration function')
ax[2].set_xlim(zoom_beat)
ax[2].set_ylim([0.25, 2.1])
ax[2].set_xlabel('Time (beats)')
ax[2].set_ylabel('Time (seconds)')
plt.tight_layout()

# %% [markdown]
# From the **beat-duration function**, it is straightforward to obtain a **beat-tempo
# function** by taking the reciprocal of the duration values multiplied with 60 (to get
# beats per minute). This function yields what we simply refer to as **tempo curve**.

# %%
def plot_tempo_curve(f_tempo, t_beat, ax=None, figsize=(8, 2), color='k', logscale=False,
                     xlabel='Time (beats)', ylabel='Temp (BPM)', xlim=None, ylim=None,
                     label='', measure_pos=[]):
    """Plot a tempo curve

    Notebook: C3/C3S3_MusicAppTempoCurve.ipynb

    Args:
        f_tempo: Tempo curve
        t_beat: Time axis of tempo curve (given as sampled beat axis)
        ax: Plot either as figure (ax==None) or into axis (ax==True) (Default value = None)
        figsize: Size of figure (Default value = (8, 2))
        color: Color of tempo curve (Default value = 'k')
        logscale: Use linear (logscale==False) or logartihmic (logscale==True) tempo axis (Default value = False)
        xlabel: Label for x-axis (Default value = 'Time (beats)')
        ylabel: Label for y-axis (Default value = 'Temp (BPM)')
        xlim: Limits for x-axis (Default value = None)
        ylim: Limits for x-axis (Default value = None)
        label: Figure labels when plotting into axis (ax==True) (Default value = '')
        measure_pos: Plot measure positions as spefified (Default value = [])

    Returns:
        fig: figure handle
        ax: axes handle
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
    ax.plot(t_beat, f_tempo, color=color, label=label)
    ax.set_title('Tempo curve')
    if xlim is None:
        xlim = [t_beat[0], t_beat[-1]]
    if ylim is None:
        ylim = [np.min(f_tempo) * 0.9, np.max(f_tempo) * 1.1]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both')
    if logscale:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
    plot_measure(ax, measure_pos)
    return fig, ax

f_tempo = 1. / f_diff_smooth_sec * 60
plot_tempo_curve(f_tempo, t_beat, xlim=zoom_beat, ylim=[30, 120])
plot_tempo_curve(f_tempo, t_beat, xlim=zoom_beat, ylim=[30, 120], logscale=True);

# %% [markdown]
# ## Computation: Overall Procedure
#
# We now define the function `compute_tempo_curve`, which summarizes the overall
# procedure described before.

# %%
def compute_tempo_curve(score, x, Fs=22050, Fs_beat=10, N=4410, H=2205, shift=0,
                        sigma=np.array([[1, 0], [0, 1], [2, 1], [1, 2], [1, 1]]),
                        win_len_beat=4):
    """Compute a tempo curve

    Notebook: C3/C3S3_MusicAppTempoCurve.ipynb

    Args:
        score (list): Score representation
        x (np.ndarray): Audio signal
        Fs (scalar): Samping rate of audio signal (Default value = 22050)
        Fs_beat (scalar): Sampling rate for beat axis (Default value = 10)
        N (int): Window size for computing audio chromagram (Default value = 4410)
        H (int): Hop size for computing audio chromagram (Default value = 2205)
        shift (int): Cyclic chroma shift applied to audio chromagram (Default value = 0)
        sigma (np.ndarray): Step size set used for DTW
            (Default value = np.array([[1, 0], [0, 1], [2, 1], [1, 2], [1, 1]]))
        win_len_beat (float): Window length (given in beats) used for smoothing tempo curve (Default value = 4)

    Returns:
        f_tempo (np.ndarray): Tempo curve
        t_beat (np.ndarray): Time axis (given in beats)
    """

    # Compute score an audio chromagram
    X_score, t_beat = compute_score_chromagram(score, Fs_beat)
    Fs_X = Fs / H
    X = librosa.feature.chroma_stft(y=x, sr=Fs, norm=2, tuning=0, hop_length=H, n_fft=N)
    X = np.roll(X, shift, axis=0)

    # Apply DTW to compte C, D, P
    C = libfmp.c3.compute_cost_matrix(X, X_score, metric='euclidean')
    D, P = librosa.sequence.dtw(C=C, step_sizes_sigma=sigma)
    P = P[::-1, :]  # reverse P
    P_mod = compute_strict_alignment_path(P)

    # Convert path into beat-time function and interpolte
    t_path_beat = P_mod[:, 1] / Fs_beat
    f_path_sec = P_mod[:, 0] / Fs_X
    f_sec = interp1d(t_path_beat, f_path_sec, kind='linear', fill_value='extrapolate')(t_beat)

    # Compute difference and smooth with Hann window
    f_diff_sec = np.diff(f_sec) * Fs_beat
    pad = np.array([f_diff_sec[-1]])
    f_diff_sec = np.concatenate((f_diff_sec, pad))
    filt_len = int(win_len_beat * Fs_beat)
    filt_win = signal.windows.hann(filt_len)
    filt_win = filt_win / np.sum(filt_win)
    f_diff_smooth_sec = scipy.ndimage.filters.convolve(f_diff_sec, filt_win, mode='reflect')

    # Compute tempo curve
    f_tempo = 1. / f_diff_smooth_sec * 60

    return f_tempo, t_beat

# Read score
fn_score = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Schumann_Op15No7_Traeumerei_MusicXML.csv')
score = libfmp.c1.csv_to_list(fn_score)
measure_pos_beat = np.array([1, 5, 9, 13, 17, 21, 25, 29, 33])

# Read wav
Fs = 22050
fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Schumann_Op15No7_Traeumerei_HernandezRomero.wav')
x, Fs = librosa.load(fn_wav)

# Compute and plot tempo curve
f_tempo, t_beat = compute_tempo_curve(score, x)
ylim = [30, 120]
plot_tempo_curve(f_tempo, t_beat, logscale=True, ylim=ylim, measure_pos=measure_pos_beat);

# %% [markdown]
# ## Parameter: Window Length
#
# The parameter `win_len_beat` determines the length of the Hann-weighted averaging
# filter (given in beats) to smooth the beat-duration function.

# %%
ylim = [25, 200]
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
plt.title('Dependency on window length')

para_dict = {}
para_dict[0] = [2, 'k']
para_dict[1] = [4, 'b']
para_dict[2] = [6, 'g']
para_dict[3] = [8, 'c']

for n in para_dict:
    win_len_beat, color = para_dict[n]
    label='Length: %d beats' % win_len_beat
    f_tempo, t_beat = compute_tempo_curve(score, x, win_len_beat=win_len_beat)
    plot_tempo_curve(f_tempo, t_beat, ax=ax, color=color, label=label,
                         logscale=True, ylim=ylim, measure_pos=measure_pos_beat)

ax.legend(loc='upper right')
ax.set_yticks([], minor=True)
yticks = np.array([25, 30, 40, 50, 60, 80, 100, 120, 160, 200])
ax.set_yticks(yticks)
plt.tight_layout()

# %% [markdown]
# ## Parameter: Step-Size Condition
#
# The quality of the warping path is decisive for the further calculation of the tempo
# curve. Besides the input representation (in our case, chroma-based features), the DTW
# results depends on parameters such as the local cost measure and the step size condition.

# %%
ylim = [25, 120]
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
plt.title('Dependency on step size condition')

para_dict = {}
para_dict[0] = [np.array([[2, 1], [1, 2], [1, 0], [0, 1], [1, 1]]), 'k']
para_dict[1] = [np.array([[1, 0], [0, 1], [1, 1]]), 'b']
para_dict[3] = [np.array([[1, 3], [3, 1], [2, 1], [1, 2], [0, 1], [1, 1], [1, 1]]), 'g']

for n in para_dict:
    sigma, color = para_dict[n]
    label = ''.join(str(s) for s in sigma)
    f_tempo, t_beat = compute_tempo_curve(score, x, sigma=sigma)
    plot_tempo_curve(f_tempo, t_beat, ax=ax, color=color, label=label,
                         logscale=True, ylim=ylim, measure_pos=measure_pos_beat)

ax.legend(loc='upper right')
plt.tight_layout()

# %% [markdown]
# ## Dependency: Feature Resolution
#
# In the previous computations, we used a sampling rate of `Fs_beat=10` for the beat-based
# reference axis of the score representation. Changing `Fs_beat` may result in substantial
# differences in the warping path and the derived tempo curve.

# %%
ylim = [20, 120]
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
plt.title('Dependency on feature resolutions')

para_dict = {}
para_dict[0] = [10, 2205, 'k']
para_dict[1] = [20, 2205, 'b']
para_dict[2] = [25, 882, 'g']

for n in para_dict:
    Fs_beat, H, color = para_dict[n]
    label='Fs_beat = %d, Fs_X = %.1f' % (Fs_beat, Fs/H)
    f_tempo, t_beat = compute_tempo_curve(score, x, Fs_beat=Fs_beat, H=H)
    plot_tempo_curve(f_tempo, t_beat, ax=ax, color=color, label=label,
                         logscale=True, ylim=ylim, measure_pos=measure_pos_beat)
ax.legend(loc='upper right')
plt.tight_layout()

# %% [markdown]
# ## Chromagram: Audio Recordings
#
# In the following experiments, we use four different recordings of the beginning
# (8 measures, 33 beats) of Robert Schumann's Träumerei.

# %%
Fs = 22050
fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Schumann_Op15No7_Traeumerei_ColliardLigoratti.wav')
x, Fs = librosa.load(fn_wav, Fs)

# Compute and plot chromagram
N = 4410
H = 2205
Fs_X = Fs / 2205
X = librosa.feature.chroma_stft(y=x, sr=Fs, norm=2, tuning=0, hop_length=H, n_fft=N)
figsize = (8, 2)
libfmp.b.plot_chromagram(X, figsize=figsize, xlabel='Time (frames)', clim=[0, 1],
                         title='Chromagram for Colliard recording', chroma_yticks=[0, 4, 7, 11])

X = np.roll(X, -5, axis=0)
libfmp.b.plot_chromagram(X, figsize=figsize, xlabel='Time (frames)', clim=[0, 1],
                         title='Chromagram for Colliard recording shifted by five semitones',
                         chroma_yticks=[0, 4, 7, 11]);

# %% [markdown]
# In the next code cell, we compute and compare the tempo curves for the four recordings.

# %%
ylim = [28, 120]
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
plt.title('Dependency on performer')

para_dict = {}
para_dict[0] = ['Romero', 'k', 0,
                os.path.join('..', 'data', 'C3', 'FMP_C3S3_Schumann_Op15No7_Traeumerei_HernandezRomero.wav')]
para_dict[1] = ['Zbinden', 'b', 0,
                os.path.join('..', 'data', 'C3', 'FMP_C3S3_Schumann_Op15No7_Traeumerei_Zbinden.wav')]
para_dict[2] = ['Ko', 'g', 0,
                os.path.join('..', 'data', 'C3', 'FMP_C3S3_Schumann_Op15No7_Traeumerei_TimKo.wav')]
para_dict[3] = ['Colliard', 'c', -5,
                os.path.join('..', 'data', 'C3', 'FMP_C3S3_Schumann_Op15No7_Traeumerei_ColliardLigoratti.wav')]

for n in para_dict:
    performer, color, shift, fn_wav = para_dict[n]
    Fs = 22050
    x, Fs = librosa.load(fn_wav, Fs)
    win_len_beat = 6
    Fs_beat = 10
    f_tempo, t_beat = compute_tempo_curve(score, x, win_len_beat=win_len_beat, Fs_beat=Fs_beat)
    plot_tempo_curve(f_tempo, t_beat, ax=ax, color=color, label=performer,
                         logscale=True, ylim=ylim, measure_pos=measure_pos_beat)
ax.legend(loc='upper right')
plt.tight_layout()

# %% [markdown]
# ## Further Notes
#
# In this notebook, we discussed how to derive tempo curves from audio recordings using
# music synchronization techniques, where the actual performance is compared with some
# score-like reference representation of the underlying musical piece. We have seen that
# the overall approach involves many design choices and parameter settings:
#
# * Feature representation (e.g., chroma features, feature resolution)
# * Alignment procedure (e.g., DTW with specific parameter settings)
# * Derivation of tempo information (e.g., differentiation, smoothing technique, filter length)
# * Interpretation of tempo values (e.g., centered view, causal view, logarithmic axis)

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Frank Zalkow.

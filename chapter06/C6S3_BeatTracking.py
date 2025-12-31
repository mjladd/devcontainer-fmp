# %% [markdown]
# # Beat Tracking by Dynamic Programming
#
# Following Section 6.3.2 of [Müller, FMP, Springer 2015], we introduce in this notebook a
# beat tracking procedure based on dynamic programming. This algorithm was originally
# described by Dan Ellis.

# %% [markdown]
# ## Problem Modelling
#
# There are many types of music with a strong and steady beat, where the tempo is more or less
# constant throughout the entire recording. We now describe a beat tracking procedure which is
# based on the assumptions that beat positions go along with the strongest note onsets and that
# the tempo is roughly constant. The main idea is to construct a score function that measures
# how well an arbitrary beat sequence reflects these two assumptions. The score-maximizing
# beat sequence constitutes the final beat tracking result.
#
# The input of the beat tracking procedure consists of a novelty function
# $\Delta:[1:N]\to\mathbb{R}$ as well as a rough estimate $\hat{\tau}\in\mathbb{R}_{>0}$ of
# the global tempo. From $\hat{\tau}$ and the feature rate, one can derive an estimate for the
# beat period. Let $\hat{\delta}\in\mathbb{N}$ be this number. Assuming a roughly constant
# tempo, the difference $\delta$ of two consecutive beats should be close to $\hat{\delta}$.
# To account for the deviation of $\delta$ from the ideal beat period $\hat{\delta}$, we
# introduce a **penalty function** $P_{\hat{\delta}}:\mathbb{N}\to\mathbb{R}$ by setting
#
# $$P_{\hat{\delta}}(\delta) := - \big( \log_2 (\delta/\hat{\delta}) \big)^2$$

# %%
import numpy as np
import os, sys, librosa
from scipy import signal
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import IPython.display as ipd
import pandas as pd

sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c6
%matplotlib inline

def compute_penalty(N, beat_ref):
    """| Compute penalty funtion used for beat tracking [FMP, Section 6.3.2]
    | Note: Concatenation of '0' because of Python indexing conventions

    Notebook: C6/C6S3_BeatTracking.ipynb

    Args:
        N (int): Length of vector representing penalty function
        beat_ref (int): Reference beat period (given in samples)

    Returns:
        penalty (np.ndarray): Penalty function
    """
    t = np.arange(1, N) / beat_ref
    penalty = -np.square(np.log2(t))
    t = np.concatenate((np.array([0]), t))
    penalty = np.concatenate((np.array([0]), penalty))
    return penalty

beat_ref = 20
N = 4 * beat_ref
penalty = compute_penalty(N, beat_ref)

plt.figure(figsize=(5, 2))
t = np.arange(0, N) / beat_ref
plt.plot(t, penalty, 'r')
plt.ylim([-4, 0.2])
plt.xlim([t[1], t[-1]])
plt.xlabel('Beat interval (given in multiples of $\hat{\delta}$)')
plt.title('Penalty')
plt.show()

# %% [markdown]
# ## Dynamic Programming Algorithm
#
# The number of possible beat sequences is exponential in $N$. However, one can compute an
# optimal beat sequence efficiently using **dynamic programming**. We have already encountered
# this algorithmic paradigm when computing a cost-minimizing warping path (DTW algorithm) or
# a probability-maximizing state sequence (Viterbi algorithm). The main idea is to break down
# the optimization problem into simpler subproblems by considering prefixes.

# %% [markdown]
# ## Implementation

# %%
def compute_beat_sequence(novelty, beat_ref, penalty=None, factor=1.0, return_all=False):
    """| Compute beat sequence using dynamic programming [FMP, Section 6.3.2]
    | Note: Concatenation of '0' because of Python indexing conventions

    Notebook: C6/C6S3_BeatTracking.ipynb

    Args:
        novelty (np.ndarray): Novelty function
        beat_ref (int): Reference beat period
        penalty (np.ndarray): Penalty function (Default value = None)
        factor (float): Weight parameter for adjusting the penalty (Default value = 1.0)
        return_all (bool): Return details (Default value = False)

    Returns:
        B (np.ndarray): Optimal beat sequence
        D (np.ndarray): Accumulated score
        P (np.ndarray): Maximization information
    """
    N = len(novelty)
    if penalty is None:
        penalty = compute_penalty(N, beat_ref)
    penalty = penalty * factor
    novelty = np.concatenate((np.array([0]), novelty))
    D = np.zeros(N+1)
    P = np.zeros(N+1, dtype=int)
    D[1] = novelty[1]
    P[1] = 0
    # forward calculation
    for n in range(2, N+1):
        m_indices = np.arange(1, n)
        scores = D[m_indices] + penalty[n-m_indices]
        maxium = np.max(scores)
        if maxium <= 0:
            D[n] = novelty[n]
            P[n] = 0
        else:
            D[n] = novelty[n] + maxium
            P[n] = np.argmax(scores) + 1
    # backtracking
    B = np.zeros(N, dtype=int)
    k = 0
    B[k] = np.argmax(D)
    while P[B[k]] != 0:
        k = k+1
        B[k] = P[B[k-1]]
    B = B[0:k+1]
    B = B[::-1]
    B = B - 1
    if return_all:
        return B, D, P
    else:
        return B

# Example from Exercise 6.12 of [Müller, FMP, Springer 2015]
nov = np.array([0.1, 0.0, 1.0, 0.0, 1.0, 0.8, 0.0, 0.2, 0.4, 1.0, 0.0])
# Manually specified penality values (in practice, we use the function "compute_penalty")
penalty = np.array([0, -2, -0.2, 1.0, 0.5, -0.1, -1, -1.5, -3, -5, -8])
factor  = 1
beat = 3

B,D,P = compute_beat_sequence(nov, beat, penalty=penalty, factor=1, return_all=True)
df = pd.DataFrame([np.arange(1,len(nov)+1), nov, D[1:], P[1:]])
df.rename(index={0:'$n$', 1:'$\Delta(n)$',
                 2:'$\mathbf{D}(n)$', 3:'$\mathbf{P}(n)$'}, inplace=True)
df.rename_axis('$n$', axis='rows')

class Formatter():
    """Cass for converting column to row format
    Notebook: C6/C6S3_BeatTracking.ipynb"""
    def __init__(self):
        self.i = 0
    def formatter(self, s):
        if self.i == 0 or self.i == 3:
            return_s = str(int(s))
        else:
            return_s = str(s)
        self.i += 1
        return return_s

ipd.display(ipd.HTML(df.to_html(formatters={i: Formatter().formatter for i in df.columns},
                                escape=False, header=False)))

print('Optimal beat sequency B:', B+1)

# %% [markdown]
# ## Example: Shostakovich
#
# We now consider a real music signal continuing with our Shostakovich example.

# %%
def beat_period_to_tempo(beat, Fs):
    """Convert beat period (samples) to tempo (BPM) [FMP, Section 6.3.2]

    Notebook: C6/C6S3_BeatTracking.ipynb

    Args:
        beat (int): Beat period (samples)
        Fs (scalar): Sample rate

    Returns:
        tempo (float): Tempo (BPM)
    """
    tempo = 60 / (beat / Fs)
    return tempo

def compute_plot_sonify_beat(x, Fs, nov, Fs_nov, beat_ref, factor, title=None, figsize=(6, 2)):
    """Compute, plot, and sonify beat sequence from novelty function [FMP, Section 6.3.2]

    Notebook: C6/C6S3_BeatTracking.ipynb

    Args:
        x: Novelty function
        Fs: Sample rate
        nov: Novelty function
        Fs_nov: Rate of novelty function
        beat_ref: Reference beat period
        factor: Weight parameter for adjusting the penalty
        title: Title of figure (Default value = None)
        figsize: Size of figure (Default value = (6, 2))
    """
    B = compute_beat_sequence(nov, beat_ref=beat_ref, factor=factor)

    beats = np.zeros(len(nov))
    beats[np.array(B, dtype=np.int32)] = 1
    if title is None:
        tempo = beat_period_to_tempo(beat_ref, Fs_nov)
        title = (r'Optimal beat sequence ($\hat{\delta}=%d$, $F_\mathrm{s}=%d$, '
                 r'$\hat{\tau}=%0.0f$ BPM, $\lambda=%0.2f$)' % (beat_ref, Fs_nov, tempo, factor))

    fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, color='k', title=title, figsize=figsize)
    T_coef = np.arange(nov.shape[0]) / Fs_nov
    ax.plot(T_coef, beats, ':r', linewidth=1)
    plt.show()

    beats_sec = T_coef[B]
    x_peaks = librosa.clicks(beats_sec, sr=Fs, click_freq=1000, length=len(x))
    ipd.display(ipd.Audio(x + x_peaks, rate=Fs))

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F07_Shostakovich_Waltz-02-Section_IncreasingTempo.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512, gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

compute_plot_sonify_beat(x, Fs, nov, Fs_nov, beat_ref=25, factor=1, figsize=(8,1.5))
compute_plot_sonify_beat(x, Fs, nov, Fs_nov, beat_ref=75, factor=0.5, figsize=(8,1.5))

# %% [markdown]
# ## Limitations
#
# The main limitation of the beat tracking procedure is its dependency on a single, predefined
# tempo $\hat{\tau}$. Using a small weighting parameter $\lambda$, the procedure may yield
# good beat tracking results even in the presence of local deviations from the ideal beat
# period $\hat{\delta}$. However, the presented procedure is not designed for handling music
# with slowly varying tempo (such as ritardando or accelerando) or abrupt changes in tempo.
# Despite these limitations, the simplicity and efficiency of the dynamic programming approach
# to beat tracking makes it an attractive choice for many types of music.

# %%
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F19_Brahms_Ormandy_sec35-53.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512, gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

beat_ref, factor = 40, 1
tempo = beat_period_to_tempo(beat_ref, Fs_nov)
compute_plot_sonify_beat(x, Fs, nov, Fs_nov, beat_ref=beat_ref, factor=factor, figsize=(8, 1.5))

beat_ref, factor = 40, 0.1
tempo = beat_period_to_tempo(beat_ref, Fs_nov)
compute_plot_sonify_beat(x, Fs, nov, Fs_nov, beat_ref=beat_ref, factor=factor, figsize=(8, 1.5))

beat_ref, factor = 80, 1
tempo = beat_period_to_tempo(beat_ref, Fs_nov)
compute_plot_sonify_beat(x, Fs, nov, Fs_nov, beat_ref=beat_ref, factor=factor, figsize=(8, 1.5))

# %% [markdown]
# ## Further Notes
#
# For modern pop and rock music with a strong beat and relatively steady tempo, the above beat
# tracking procedure by Dan Ellis yields good results as long as the novelty curve reveals most
# of the relevant beat onsets. To compute an optimal beat sequence, we discussed an efficient
# and elegant algorithm based on dynamic programming (DP)—a paradigm we have already encountered
# in the context of dynamic time warping. The detection of periodic beat patterns becomes
# challenging when the music recording reveals significant tempo changes. In the notebook on
# predominant pulse tracking (PLP), we studied an alternative approach that aims at detecting
# locally periodic patterns such that even sudden tempo changes may be captured.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

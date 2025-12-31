# %% [markdown]
# # Phase-Based Novelty
#
# Following Section 6.1.3 of [Müller, FMP, Springer 2015], we introduce in this notebook
# a phase-based approach for computing a novelty function.

# %% [markdown]
# ## Phase Representation
#
# Stationary tones have a stable phase, while transients have an unstable phase. Using polar
# coordinate representation:
#
# $$\mathcal{X}(n,k)= |\mathcal{X}(n,k)| \,\,\mathrm{exp}(2\pi i\varphi(n,k))$$
#
# For locally stationary signals, the frame-wise phase difference remains approximately constant:
#
# $$\varphi(n,k)- \varphi(n-1,k)  \approx \varphi(n-1,k)- \varphi(n-2,k)$$

# %% [markdown]
# ## Principal Argument Function
#
# When considering phase differences, we need to handle phase wrapping discontinuities using
# the principal argument function:
#
# $$\Psi:\mathbb{R}\to\left[-0.5,0.5\right]$$

# %%
import numpy as np
import os, sys, librosa
from scipy import signal
from matplotlib import pyplot as plt
import IPython.display as ipd

sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c6
from libfmp.c6 import compute_local_average

%matplotlib inline

def principal_argument(v):
    """Principal argument function

    | Notebook: C6/C6S1_NoveltyPhase.ipynb, see also
    | Notebook: C8/C8S2_InstantFreqEstimation.ipynb

    Args:
        v (float or np.ndarray): Value (or vector of values)

    Returns:
        w (float or np.ndarray): Principle value of v
    """
    w = np.mod(v + 0.5, 1) - 0.5
    return w

v = np.arange(-1,2,0.01)

plt.figure(figsize=(6,2))
plt.plot(v, principal_argument(v), 'r')
plt.title(r'Principle argument function $\Psi$')
plt.xlabel(r'$v$')
plt.ylabel(r'$\Psi(v)$')
plt.xlim([v[0], v[-1]])
plt.tight_layout()

# %% [markdown]
# ## Phase-Based Novelty Function
#
# Using the principal argument function, we define the first-order and second-order phase
# differences:
#
# $$\varphi'(n,k) := \Psi\big(\varphi(n,k)- \varphi(n-1,k)\big)$$
# $$\varphi''(n,k) := \Psi\big(\varphi'(n,k)- \varphi'(n-1,k)\big)$$
#
# The phase-based novelty function is:
#
# $$\Delta_\mathrm{Phase}(n) = \sum_{k=0}^{K} |\varphi''(n,k)|$$

# %% [markdown]
# ## Implementation

# %%
def compute_novelty_phase(x, Fs=1, N=1024, H=64, M=40, norm=True):
    """Compute phase-based novelty function

    Notebook: C6/C6S1_NoveltyPhase.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = 1024)
        H (int): Hop size (Default value = 64)
        M (int): Determines size (2M+1) in samples of centric window  used for local average (Default value = 40)
        norm (bool): Apply max norm (if norm==True) (Default value = True)

    Returns:
        novelty_phase (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Fs_feature = Fs / H
    phase = np.angle(X) / (2*np.pi)
    phase_diff = principal_argument(np.diff(phase, axis=1))
    phase_diff2 = principal_argument(np.diff(phase_diff, axis=1))
    novelty_phase = np.sum(np.abs(phase_diff2), axis=0)
    novelty_phase = np.concatenate((novelty_phase, np.array([0, 0])))
    if M > 0:
        local_average = compute_local_average(novelty_phase, M)
        novelty_phase = novelty_phase - local_average
        novelty_phase[novelty_phase < 0] = 0
    if norm:
        max_value = np.max(novelty_phase)
        if max_value > 0:
            novelty_phase = novelty_phase / max_value
    return novelty_phase, Fs_feature

fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_F01_Queen.csv')
ann, label_keys = libfmp.c6.read_annotation_pos(fn_ann)

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F01_Queen.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)
x_duration = len(x)/Fs

nov, Fs_nov = compute_novelty_phase(x, Fs=Fs, M=0, norm=0)
libfmp.b.plot_signal(nov, Fs_nov, color='k',
    title='Phase-based novelty function');

nov, Fs_nov = compute_novelty_phase(x, Fs=Fs, M=10, norm=1)
fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, color='k',
                    title='Phase-based novelty function with post-processing')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                    nontime_axis=True, time_min=0, time_max=x_duration);

# %% [markdown]
# ## Role of Hopsize Parameter
#
# Phase-based approaches are quite sensitive with regard to the hopsize. Using a small
# hopsize is often beneficial.

# %%
H_set = [256, 128, 64]

for H in H_set:
    nov, Fs_nov = compute_novelty_phase(x, Fs=Fs, N=1024, H=H, M=10, norm=1)
    fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, color='k',
                        title='Phase-based novelty function (H=%d) with post-processing'%H)
    libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                        nontime_axis=True, time_min=0, time_max=x_duration);

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Angel Villar-Corrales.

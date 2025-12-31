# %% [markdown]
# # Spectral-Based Novelty
#
# Following Section 6.1.2 of [Müller, FMP, Springer 2015], we introduce in this notebook
# a spectral-based approach for computing a novelty function, which is also often referred
# to as spectral flux.

# %% [markdown]
# ## Spectral-Based Novelty
#
# The idea of spectral-based novelty detection is to:
# * First convert the signal into a time-frequency representation
# * Then capture changes in the frequency content
#
# This results in a spectral-based novelty function, also known as the spectral flux.

# %% [markdown]
# ## Logarithmic Compression
#
# To enhance weak spectral components, we apply logarithmic compression:
#
# $$\mathcal{Y}:=\Gamma_\gamma(|\mathcal{X}|)=\log(1+ \gamma \cdot |\mathcal{X}|)$$

# %%
import os, sys
import numpy as np
from numba import jit
import librosa
from scipy import signal
from matplotlib import pyplot as plt
import IPython.display as ipd

sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c6

%matplotlib inline

fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_F01_Queen.csv')
ann, label_keys = libfmp.c6.read_annotation_pos(fn_ann)

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F01_Queen.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)
x_duration = len(x)/Fs
N, H = 1024, 256
X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')

fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.05], 'height_ratios': [1, 1, 1]}, figsize=(6.5, 6))

Y = np.abs(X)
libfmp.b.plot_matrix(Y, Fs=Fs/H, Fs_F=N/Fs, ax=[ax[0,0], ax[0,1]], title='No compression')

gamma = 1
Y = np.log(1 + gamma * np.abs(X))
libfmp.b.plot_matrix(Y, Fs=Fs/H, Fs_F=N/Fs, ax=[ax[1,0], ax[1,1]], title='Compression using $\gamma=%0.0f$'%gamma)

gamma = 100
Y = np.log(1 + gamma * np.abs(X))
libfmp.b.plot_matrix(Y, Fs=Fs/H, Fs_F=N/Fs, ax=[ax[2,0], ax[2,1]], title='Compression using $\gamma=%0.0f$'%gamma)
plt.tight_layout()

# %% [markdown]
# ## Discrete Derivative, Half-Wave Rectification, Accumulation
#
# We compute the discrete temporal derivative of the compressed spectrum, apply half-wave
# rectification, and sum over the frequency axis:
#
# $$\Delta_\mathrm{Spectral}(n):= \sum_{k=0}^K  \big|\mathcal{Y}(n+1,k)-\mathcal{Y}(n,k)\big|_{\geq 0}$$

# %%
Y = np.log(1 + 100 * np.abs(X))
Y_diff = np.diff(Y, n=1)
Y_diff[Y_diff < 0] = 0
nov = np.sum(Y_diff, axis=0)
nov = np.concatenate((nov, np.array([0])))
Fs_nov = Fs/H

fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, color='k', title='Spectral-based novelty function')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,  time_min=0, time_max=x_duration);

# %% [markdown]
# ## Subtracting Local Average
#
# We introduce a local average function and subtract it to enhance the peak structure:
#
# $$\mu(n):= \frac{1}{2M+1}\sum_{m=-M}^M  \Delta_\mathrm{Spectral}(n+m)$$
#
# $$\bar{\Delta}_\mathrm{Spectral}(n):= \big|\Delta_\mathrm{Spectral}(n)-\mu(n)\big|_{\geq 0}$$

# %%
@jit(nopython=True)
def compute_local_average(x, M):
    """Compute local average of signal

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x (np.ndarray): Signal
        M (int): Determines size (2M+1) in samples of centric window  used for local average

    Returns:
        local_average (np.ndarray): Local average signal
    """
    L = len(x)
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average

M_sec = 0.1
M = int(np.ceil(M_sec * Fs_nov))

local_average = compute_local_average(nov, M)
nov_norm =  nov - local_average
nov_norm[nov_norm<0]=0
nov_norm = nov_norm / max(nov_norm)

libfmp.b.plot_signal(nov, Fs_nov, color='k',
    title='Novelty function with local average curve');

t_novelty = np.arange(nov.shape[0]) / Fs_nov
plt.plot(t_novelty, local_average, 'r')
plt.tight_layout()

fig, ax, line = libfmp.b.plot_signal(nov_norm, Fs_nov, color='k',
                    title='Novelty function after local normalization')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                    nontime_axis=True, time_min=0, time_max=x_duration);

# %% [markdown]
# ## Implementation

# %%
def compute_novelty_spectrum(x, Fs=1, N=1024, H=256, gamma=100.0, M=10, norm=True):
    """Compute spectral-based novelty function

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = 1024)
        H (int): Hop size (Default value = 256)
        gamma (float): Parameter for logarithmic compression (Default value = 100.0)
        M (int): Size (frames) of local average (Default value = 10)
        norm (bool): Apply max norm (if norm==True) (Default value = True)

    Returns:
        novelty_spectrum (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Fs_feature = Fs / H
    Y = np.log(1 + gamma * np.abs(X))
    Y_diff = np.diff(Y)
    Y_diff[Y_diff < 0] = 0
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    return novelty_spectrum, Fs_feature

nov, Fs_nov = compute_novelty_spectrum(x, Fs=Fs, N=1024, H=256, gamma=10, M=10, norm=True)
fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, color='k',
                    title='Spectral-based novelty function with normalization')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                    nontime_axis=True, time_min=0, time_max=x_duration);

nov, Fs_nov = libfmp.c6.compute_novelty_energy(x, Fs=Fs, N=1024, H=256, gamma=10)
fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, color='k',
                    title='Energy-based novelty function with normalization')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                    nontime_axis=True, time_min=0, time_max=x_duration);

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Angel Villar-Corrales.

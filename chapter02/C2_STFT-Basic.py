# %% [markdown]
# # Discrete Short-Time Fourier Transform (STFT)
#
# Following Section 2.1.4 of [Muller, FMP, Springer 2015], we discuss in this
# notebook a discrete version of the short-time Fourier transform (STFT).

# %% [markdown]
# ## Missing Time Localization
#
# The Fourier transform yields frequency information that is averaged over the
# entire time domain. However, the information on **when** these frequencies
# occur is hidden in the transform.

# %%
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import librosa
from ipywidgets import interact, fixed, FloatSlider
import IPython.display as ipd
%matplotlib inline

Fs = 128
duration = 10
omega1 = 1
omega2 = 5
N = int(duration * Fs)
t = np.arange(N) / Fs
t1 = t[:N // 2]
t2 = t[N // 2:]

x1 = 1.0 * np.sin(2 * np.pi * omega1 * t1)
x2 = 0.7 * np.sin(2 * np.pi * omega2 * t2)
x = np.concatenate((x1, x2))

plt.figure(figsize=(8, 2))
plt.subplot(1, 2, 1)
plt.plot(t, x, c='k')
plt.xlim([min(t), max(t)])
plt.xlabel('Time (seconds)')

plt.subplot(1, 2, 2)
X = np.abs(np.fft.fft(x)) / Fs
freq = np.fft.fftfreq(N, d=1 / Fs)
X = X[:N // 2]
freq = freq[:N // 2]
plt.plot(freq, X, c='k')
plt.xlim([0, 7])
plt.ylim([0, 3])
plt.xlabel('Frequency (Hz)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Basic Idea
#
# To recover the hidden time information, Dennis Gabor introduced in 1946 the
# **short-time Fourier transform** (STFT). Instead of considering the entire signal,
# the main idea is to consider only a small section of the signal using a
# **window function**. The original signal is then multiplied with the window function
# to yield a **windowed signal**. To obtain frequency information at different time
# instances, one shifts the window function across time and computes a Fourier
# transform for each of the resulting windowed signals.

# %%
def windowed_ft(t, x, Fs, w_pos_sec, w_len):
    N = len(x)
    w_pos = int(Fs * w_pos_sec)
    w_padded = np.zeros(N)
    w_padded[w_pos:w_pos + w_len] = 1
    x = x * w_padded
    plt.figure(figsize=(8, 2))

    plt.subplot(1, 2, 1)
    plt.plot(t, x, c='k')
    plt.plot(t, w_padded, c='r')
    plt.xlim([min(t), max(t)])
    plt.ylim([-1.1, 1.1])
    plt.xlabel('Time (seconds)')

    plt.subplot(1, 2, 2)
    X = np.abs(np.fft.fft(x)) / Fs
    freq = np.fft.fftfreq(N, d=1 / Fs)
    X = X[:N // 2]
    freq = freq[:N // 2]
    plt.plot(freq, X, c='k')
    plt.xlim([0, 7])
    plt.ylim([0, 3])
    plt.xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()


w_len = 4 * Fs
windowed_ft(t, x, Fs, w_pos_sec=1, w_len=w_len)
windowed_ft(t, x, Fs, w_pos_sec=3, w_len=w_len)
windowed_ft(t, x, Fs, w_pos_sec=5, w_len=w_len)

# %% [markdown]
# ## Formal Definition of the Discrete STFT
#
# Let x:[0:L-1] -> R be a real-valued discrete-time signal of length L.
# Let w:[0:N-1] -> R be a sampled window function of length N.
# The length parameter N determines the duration of the considered sections (N/Fs seconds).
# The hop size H determines the step size in which the window is shifted.
#
# The **discrete STFT** X of the signal x is given by:
# X(m,k) := sum_{n=0}^{N-1} x(n+mH) * w(n) * exp(-2*pi*i*k*n/N)
#
# with m in [0:M] and k in [0:K], where M := floor((L-N)/H) is the maximal frame index
# and K = N/2 is the frequency index corresponding to the Nyquist frequency.

# %% [markdown]
# ## Spectrogram
#
# The **spectrogram** is a two-dimensional representation of the squared magnitude
# of the STFT: Y(m,k) := |X(m,k)|^2
#
# It can be visualized as a 2D image where the horizontal axis represents time
# and the vertical axis represents frequency.

# %%
def stft_basic(x, w, H=8, only_positive_frequencies=False):
    """Compute a basic version of the discrete short-time Fourier transform (STFT)

    Args:
        x (np.ndarray): Signal to be transformed
        w (np.ndarray): Window function
        H (int): Hopsize (Default value = 8)
        only_positive_frequencies (bool): Return only positive frequency part

    Returns:
        X (np.ndarray): The discrete short-time Fourier transform
    """
    N = len(w)
    L = len(x)
    M = np.floor((L - N) / H).astype(int) + 1
    X = np.zeros((N, M), dtype='complex')
    for m in range(M):
        x_win = x[m * H:m * H + N] * w
        X_win = np.fft.fft(x_win)
        X[:, m] = X_win

    if only_positive_frequencies:
        K = 1 + N // 2
        X = X[0:K, :]
    return X


H = 8
N = 128
w = np.ones(N)
X = stft_basic(x, w, H, only_positive_frequencies=True)
Y = np.abs(X) ** 2

plt.figure(figsize=(8, 2))
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(t)), x, c='k')
plt.xlim([0, len(t)])
plt.xlabel('Index (samples)')
plt.subplot(1, 2, 2)
plt.imshow(Y, origin='lower', aspect='auto', cmap='gray_r')
plt.xlabel('Index (frames)')
plt.ylabel('Index (frequency)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Interpretation of Time and Frequency Indices
#
# Each Fourier coefficient X(m,k) is associated with the physical time position:
# T_coef(m) := m * H / Fs (in seconds)
#
# The index k corresponds to the physical frequency:
# F_coef(k) := k * Fs / N (in Hertz)

# %%
T_coef = np.arange(X.shape[1]) * H / Fs
F_coef = np.arange(X.shape[0]) * Fs / N

plt.figure(figsize=(8, 2))

plt.subplot(1, 2, 1)
plt.plot(t, x, c='k')
plt.xlim([min(t), max(t)])
plt.xlabel('Time (seconds)')

plt.subplot(1, 2, 2)
left = min(T_coef)
right = max(T_coef) + N / Fs
lower = min(F_coef)
upper = max(F_coef)
plt.imshow(Y, origin='lower', aspect='auto', cmap='gray_r',
           extent=[left, right, lower, upper])
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Example: C-major Scale on Piano
#
# The spectrogram reveals the frequency information of played notes over time.
# For each note, horizontal lines stacked on top of each other correspond to
# the partials (integer multiples of the fundamental frequency).

# %%
fn_wav = os.path.join('..', 'data', 'C2', 'FMP_C2_F10.wav')
x, Fs = librosa.load(fn_wav)

H = 1024
N = 2048
w = np.hanning(N)
X = stft_basic(x, w, H)
Y = np.abs(X) ** 2
eps = np.finfo(float).eps
Y_db = 10 * np.log10(Y + eps)

T_coef = np.arange(X.shape[1]) * H / Fs
F_coef = np.arange(X.shape[0]) * Fs / N

fig = plt.figure(figsize=(8, 5))

gs = matplotlib.gridspec.GridSpec(3, 2, height_ratios=[1, 2, 2], width_ratios=[100, 2])
ax1, ax2, ax3, ax4, ax5, ax6 = [plt.subplot(gs[i]) for i in range(6)]

t = np.arange(len(x)) / Fs
ax1.plot(t, x, c='gray')
ax1.set_xlim([min(t), max(t)])

ax2.set_visible(False)

left = min(T_coef)
right = max(T_coef) + N / Fs
lower = min(F_coef)
upper = max(F_coef)

im1 = ax3.imshow(Y, origin='lower', aspect='auto', cmap='gray_r',
                 extent=[left, right, lower, upper])
im1.set_clim([0, 1000])
ax3.set_ylim([0, 5000])
ax3.set_ylabel('Frequency (Hz)')
cbar = fig.colorbar(im1, cax=ax4)
ax4.set_ylabel('Magnitude (linear)', rotation=90)

im2 = ax5.imshow(Y_db, origin='lower', aspect='auto', cmap='gray_r',
                 extent=[left, right, lower, upper])
im2.set_clim([-30, 20])
ax5.set_ylim([0, 5000])
ax5.set_xlabel('Time (seconds)')
ax5.set_ylabel('Frequency (Hz)')
cbar = fig.colorbar(im2, cax=ax6)
ax6.set_ylabel('Magnitude (dB)', rotation=90)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Further Notes
#
# The libfmp library provides an implementation of the STFT including padding options
# and the centric view as used in the FMP notebooks.

# %%
import sys
sys.path.append('..')
import libfmp.c2

fn_wav = os.path.join('..', 'data', 'C2', 'FMP_C2_F10.wav')
x, Fs = librosa.load(fn_wav)

H = 1024
N = 2048
w = np.hanning(N)
X = libfmp.c2.stft(x, w, H)
Y = np.abs(X) ** 2
eps = np.finfo(float).eps
Y_db = 10 * np.log10(Y + eps)

T_coef = np.arange(X.shape[1]) * H / Fs
F_coef = np.arange(X.shape[0]) * Fs / N

fig = plt.figure(figsize=(8, 3))

gs = matplotlib.gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[100, 2])
ax1, ax2, ax3, ax4 = [plt.subplot(gs[i]) for i in range(4)]

t = np.arange(len(x)) / Fs
ax1.plot(t, x, c='gray')
ax1.set_xlim([min(t), max(t)])

ax2.set_visible(False)

left = min(T_coef)
right = max(T_coef) + N / Fs
lower = min(F_coef)
upper = max(F_coef)

im = ax3.imshow(Y_db, origin='lower', aspect='auto', cmap='gray_r',
                extent=[left, right, lower, upper])
im.set_clim([-30, 20])
ax3.set_ylim([0, 5000])
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Frequency (Hz)')
cbar = fig.colorbar(im, cax=ax4)
ax4.set_ylabel('Magnitude (dB)', rotation=90)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller and Frank Zalkow.

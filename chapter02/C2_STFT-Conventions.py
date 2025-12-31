# %% [markdown]
# # STFT: Conventions and Implementations
#
# In this notebook, we summarize the various variants for computing and interpreting
# a discrete STFT, while fixing the conventions used throughout the FMP notebooks.

# %% [markdown]
# ## Time Axis Conventions for Sampled Signals
#
# Let x = (x(0), x(1), ..., x(L-1)) in R^L be a discrete-time signal of length L.
# Let Fs be the sample rate. We associate physical time positions (in seconds):
# t(n) = n / Fs
#
# - Sample x(0) is associated to physical time t(0) = 0 seconds
# - Duration of signal x is L / Fs seconds
# - Sampling period is 1 / Fs

# %%
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import librosa
import librosa.display
import IPython.display as ipd

sys.path.append('..')
import libfmp.b
import libfmp.c2

%matplotlib inline

# Load wav
fn_wav = os.path.join('..', 'data', 'C2', 'FMP_C2_F05c_C4_violin.wav')

Fs = 11025
x, Fs = librosa.load(fn_wav, sr=Fs)
ipd.Audio(x, rate=Fs)
L = x.shape[0]
t_wav = np.arange(L) / Fs
x_duration = L / Fs

print('t[0] = %0.4f, t[-1] = (L-1)/Fs = %0.4f, Fs = %0.0f, L = %0.0f, dur_x=%0.4f'
      % (t_wav[0], t_wav[-1], Fs, L, x_duration))
ipd.display(ipd.Audio(x, rate=Fs))

plt.figure(figsize=(6, 2))
plt.plot(t_wav, x, color='gray')
plt.xlim([t_wav[0], t_wav[-1]])
plt.xlabel('Time (seconds)')
plt.tight_layout()
plt.show()

# %%
# Using librosa.display.waveshow for envelope visualization
plt.figure(figsize=(6, 2))
librosa.display.waveshow(x, sr=Fs, color='gray')
plt.tight_layout()
plt.show()

# %%
# Using libfmp.b.plot_signal
libfmp.b.plot_signal(x, Fs, figsize=(6, 2))
plt.show()

# %% [markdown]
# ## Centered Windowing and Time Conversion
#
# When computing an STFT, we adopt a **centered** view, where the center of the
# window is used as reference. We extend the signal to the left by zero padding
# of half the window length.
#
# The frame index m is associated to physical time position:
# T_coef(m) := m * H / Fs (in seconds)
#
# - Frame index m=0 corresponds to physical time T_coef(0) = 0 seconds
# - Time resolution (distance between frames) is Delta_t = H / Fs seconds

# %% [markdown]
# ## Frequency Conversion
#
# When x and w are real-valued, only coefficients k in [0:K] with K = N/2 are used.
# Index k = N/2 corresponds to the Nyquist frequency omega = Fs/2.
# Index k corresponds to frequency:
# F_coef(k) := k * Fs / N (in Hz)

# %% [markdown]
# ## Spectrogram Visualization
#
# Using librosa.stft with center=True (centered view) and pad_mode='constant'
# (zero-padding).

# %%
N = 256
H = 64
color = 'gray_r'

X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann',
                 pad_mode='constant', center=True)
Y = np.log(1 + 100 * np.abs(X) ** 2)

T_coef = np.arange(X.shape[1]) * H / Fs
T_coef_librosa = librosa.frames_to_time(np.arange(X.shape[1]), sr=Fs, hop_length=H)
print('Computation of T_coef agrees:', np.allclose(T_coef, T_coef_librosa))

K = N // 2
F_coef = np.arange(K + 1) * Fs / N
F_coef_librosa = librosa.fft_frequencies(sr=Fs, n_fft=N)
print('Computation of F_coef agrees:', np.allclose(F_coef, F_coef_librosa))

plt.figure(figsize=(6, 3))
extent = [T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]]
plt.imshow(Y, cmap=color, aspect='auto', origin='lower', extent=extent)
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.colorbar()
plt.tight_layout()
plt.show()

# %%
# Centered view in visualization: adjust margins
plt.figure(figsize=(6, 3))
extent = [T_coef[0] - (H / 2) / Fs, T_coef[-1] + (H / 2) / Fs,
          F_coef[0] - (Fs / N) / 2, F_coef[-1] + (Fs / N) / 2]
plt.imshow(Y, cmap=color, aspect='auto', origin='lower', extent=extent)
plt.xlim([T_coef[0], T_coef[-1]])
plt.ylim([F_coef[0], F_coef[-1]])
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.colorbar()
plt.tight_layout()
plt.show()

# %%
# Using librosa.display.specshow
plt.figure(figsize=(6, 3))
librosa.display.specshow(Y, y_axis='linear', x_axis='time', sr=Fs, hop_length=H, cmap=color)
plt.colorbar()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## libfmp Implementations
#
# Functions for computing and visualizing the STFT are part of libfmp.

# %%
w = np.hanning(N)
X = libfmp.c2.stft(x, w, H, zero_padding=0, only_positive_frequencies=True)
Y = np.log(1 + 100 * np.abs(X) ** 2)
print('=== Using libfmp.c2.stft ===')
print('Y.shape = (%d, %d), Y.dtype = %s' % (Y.shape[0], Y.shape[1], Y.dtype))
libfmp.b.plot_matrix(Y, Fs=Fs / H, Fs_F=N / Fs)
plt.show()

Y, T_coef, F_coef = libfmp.c2.stft_convention_fmp(x, Fs, N, H, mag=True, gamma=100)
print('=== Using libfmp.c2.stft_convention_fmp ===')
print('Y.shape = (%d, %d), Y.dtype = %s' % (Y.shape[0], Y.shape[1], Y.dtype))
libfmp.b.plot_matrix(Y, Fs=Fs / H, Fs_F=N / Fs)
plt.show()

# %% [markdown]
# ## STFT with Increased Frequency Grid Density
#
# Using zero-padding to refine the frequency grid. In librosa.stft:
# - n_fft = L (size of padded section)
# - win_length = N (size of windowed section)
#
# With this, F_coef(k) = k * Fs / L

# %%
N = 256
L = 512
H = 64
color = 'gray_r'

X = librosa.stft(x, n_fft=L, hop_length=H, win_length=N, window='hann',
                 pad_mode='constant', center=True)
Y = np.log(1 + 100 * np.abs(X) ** 2)

T_coef = np.arange(0, X.shape[1]) * H / Fs

K = L // 2
F_coef = np.arange(K + 1) * Fs / L
F_coef_librosa = librosa.fft_frequencies(sr=Fs, n_fft=L)
print('Computation of F_coef agrees:', np.allclose(F_coef, F_coef_librosa))
print('Y.shape = (%d,%d)' % (Y.shape[0], Y.shape[1]))

plt.figure(figsize=(6, 3))
extent = [T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]]
plt.imshow(Y, cmap=color, aspect='auto', origin='lower', extent=extent)
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.colorbar()
plt.tight_layout()
plt.show()

# %%
# Zero padding with libfmp: use zero_padding argument (L - N)
w = np.hanning(N)
X = libfmp.c2.stft(x, w, H, zero_padding=N, only_positive_frequencies=True)
Y = np.log(1 + 100 * np.abs(X) ** 2)

print('Y.shape = (%d,%d)' % (Y.shape[0], Y.shape[1]))
libfmp.b.plot_matrix(Y, Fs=Fs / H, Fs_F=(N + N) / Fs)
plt.show()

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller and Frank Zalkow.

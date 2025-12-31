# %% [markdown]
# # STFT: Frequency Grid Density
#
# Computing a discrete STFT introduces a frequency grid which resolution depends
# on the signal's sampling rate and the STFT window size. In this notebook, we
# discuss how to make the frequency grid denser by suitably padding the windowed
# sections in the STFT computation.
#
# **Important:** Often, one loosely says that this procedure increases the frequency
# resolution. This, however, is not true in a qualitative sense.

# %% [markdown]
# ## DFT Frequency Grid
#
# Let x in R^N be a discrete signal of length N with sampling rate Fs.
# The DFT X = DFT_N * x can be interpreted as an approximation of the continuous
# Fourier transform for certain frequencies:
#
# X(k) ~ Fs * f_hat(k * Fs / N)
#
# The index k corresponds to the physical frequency F_coef(k) = k * Fs / N (in Hz).
#
# To increase the density of the frequency grid, we can apply **zero padding**:
# Append zeros to the signal to create x_tilde of length L >= N, then apply DFT_L.
# This gives a linear frequency resolution of Fs/L instead of Fs/N.
#
# **Note:** Zero padding does NOT improve the approximation quality of the DFT.
# Only the linear sampling of the frequency axis is refined.

# %%
import numpy as np
from matplotlib import pyplot as plt
import librosa
%matplotlib inline

Fs = 32
duration = 2
freq1 = 5
freq2 = 15
N = int(duration * Fs)
t = np.arange(N) / Fs
t1 = t[:N // 2]
t2 = t[N // 2:]

x1 = 1.0 * np.sin(2 * np.pi * freq1 * t1)
x2 = 0.7 * np.sin(2 * np.pi * freq2 * t2)
x = np.concatenate((x1, x2))

plt.figure(figsize=(8, 2))

ax1 = plt.subplot(1, 2, 1)
plt.plot(x, c='k')
plt.title('Original signal ($N$=%d)' % N)
plt.xlabel('Time (samples)')
plt.xlim([0, N - 1])
plt.subplot(1, 2, 2)
Y = np.abs(np.fft.fft(x)) / Fs
plt.plot(Y, c='k')
plt.title('Magnitude DFT of original signal ($N$=%d)' % N)
plt.xlabel('Frequency (bins)')
plt.xlim([0, N - 1])
plt.tight_layout()
plt.show()

L = 2 * N
pad_len = L - N
t_tilde = np.concatenate((t, np.arange(len(x), len(x) + pad_len) / Fs))
x_tilde = np.concatenate((x, np.zeros(pad_len)))

plt.figure(figsize=(8, 2))
ax1 = plt.subplot(1, 2, 1)
plt.plot(x_tilde, c='k')
plt.title('Padded signal ($L$=%d)' % L)
plt.xlabel('Time (samples)')
plt.xlim([0, L - 1])
plt.subplot(1, 2, 2)
Y_tilde = np.abs(np.fft.fft(x_tilde)) / Fs
plt.plot(Y_tilde, c='k')
plt.title('Magnitude DFT of padded signal ($L$=%d)' % L)
plt.xlabel('Frequency (bins)')
plt.xlim([0, L - 1])

plt.tight_layout()
plt.show()

# %%
def compute_plot_DFT_extended(t, x, Fs, L):
    N = len(x)
    pad_len = L - N
    t_tilde = np.concatenate((t, np.arange(len(x), len(x) + pad_len) / Fs))
    x_tilde = np.concatenate((x, np.zeros(pad_len)))
    Y = np.abs(np.fft.fft(x_tilde)) / Fs
    Y = Y[:L // 2]
    freq = np.arange(L // 2) * Fs / L

    plt.figure(figsize=(12, 2))

    ax1 = plt.subplot(1, 3, 1)
    plt.plot(t_tilde, x_tilde, c='k')
    plt.title('Signal ($N$=%d)' % N)
    plt.xlabel('Time (seconds)')
    plt.xlim([t[0], t[-1]])

    ax2 = plt.subplot(1, 3, 2)
    plt.plot(t_tilde, x_tilde, c='k')
    plt.title('Padded signal (of size $L$=%d)' % L)
    plt.xlabel('Time (seconds)')
    plt.xlim([t_tilde[0], t_tilde[-1]])

    ax3 = plt.subplot(1, 3, 3)
    plt.plot(freq, Y, c='k')
    plt.title('Magnitude DFT of padded signal ($L$=%d)' % L)
    plt.xlabel('Frequency (Hz)')
    plt.xlim([freq[0], freq[-1]])
    plt.tight_layout()

    return ax1, ax2, ax3


N = len(x)

L = N
ax1, ax2, ax3 = compute_plot_DFT_extended(t, x, Fs, L)
plt.show()

L = 2 * N
ax1, ax2, ax3 = compute_plot_DFT_extended(t, x, Fs, L)
plt.show()

L = 4 * N
ax1, ax2, ax3 = compute_plot_DFT_extended(t, x, Fs, L)
plt.show()

# %% [markdown]
# ## STFT with Increased Frequency Grid Resolution
#
# The same zero-padding strategy can be used to increase the frequency grid
# resolution of an STFT. The `librosa.stft` function implements this with the
# parameters `n_fft` (corresponding to L) and `win_length` (corresponding to N).

# %%
import os
import IPython.display as ipd

# Load wav
fn_wav = os.path.join('..', 'data', 'C2', 'FMP_C2_F05c_C4_violin.wav')

Fs = 11025
x, Fs = librosa.load(fn_wav, sr=Fs)
ipd.display(ipd.Audio(x, rate=Fs))

t_wav = np.arange(0, x.shape[0]) * 1 / Fs
plt.figure(figsize=(5, 1.5))
plt.plot(t_wav, x, c='gray')
plt.xlim([t_wav[0], t_wav[-1]])
plt.xlabel('Time (seconds)')
plt.tight_layout()
plt.show()

# %%
def compute_stft(x, Fs, N, H, L=None, pad_mode='constant', center=True):
    if L is None:
        L = N
    X = librosa.stft(x, n_fft=L, hop_length=H, win_length=N,
                     window='hann', pad_mode=pad_mode, center=center)
    Y = np.log(1 + 100 * np.abs(X) ** 2)
    F_coef = librosa.fft_frequencies(sr=Fs, n_fft=L)
    T_coef = librosa.frames_to_time(np.arange(X.shape[1]), sr=Fs, hop_length=H)
    return Y, F_coef, T_coef


def plot_compute_spectrogram(x, Fs, N, H, L, color='gray_r'):
    Y, F_coef, T_coef = compute_stft(x, Fs, N, H, L)
    plt.imshow(Y, cmap=color, aspect='auto', origin='lower')
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (bins)')
    plt.title('L=%d' % L)
    plt.colorbar()


N = 256
H = 64
color = 'gray_r'
plt.figure(figsize=(10, 4))

L = N
plt.subplot(1, 3, 1)
plot_compute_spectrogram(x, Fs, N, H, L)

L = 2 * N
plt.subplot(1, 3, 2)
plot_compute_spectrogram(x, Fs, N, H, L)

L = 4 * N
plt.subplot(1, 3, 3)
plot_compute_spectrogram(x, Fs, N, H, L)

plt.tight_layout()
plt.show()

# %%
def plot_compute_spectrogram_physical(x, Fs, N, H, L, xlim, ylim, color='gray_r'):
    Y, F_coef, T_coef = compute_stft(x, Fs, N, H, L)
    extent = [T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]]
    plt.imshow(Y, cmap=color, aspect='auto', origin='lower', extent=extent)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.title('L=%d' % L)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.colorbar()


xlim_sec = [2, 3]
ylim_hz = [2000, 3000]

plt.figure(figsize=(10, 4))

L = N
plt.subplot(1, 3, 1)
plot_compute_spectrogram_physical(x, Fs, N, H, L, xlim=xlim_sec, ylim=ylim_hz)

L = 2 * N
plt.subplot(1, 3, 2)
plot_compute_spectrogram_physical(x, Fs, N, H, L, xlim=xlim_sec, ylim=ylim_hz)

L = 4 * N
plt.subplot(1, 3, 3)
plot_compute_spectrogram_physical(x, Fs, N, H, L, xlim=xlim_sec, ylim=ylim_hz)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller.

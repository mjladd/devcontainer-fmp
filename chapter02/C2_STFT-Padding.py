# %% [markdown]
# # STFT: Padding
#
# In this notebook, we discuss various padding strategies that become important
# when implementing and interpreting an STFT.

# %% [markdown]
# ## Padding Variants
#
# When using the STFT with window size N:
# 1. The center of the first window corresponds to sample index N/2.
#    To have it correspond to time 0, pad N/2 samples at the beginning.
# 2. The last window's time range is fully contained in the signal's time range.
#    To counteract this, pad at the end of the signal (often N/2 samples).
#
# Padding strategies:
# - **Zero padding**: Expand the signal with zeros
# - **Reflect padding**: Mirror the signal on the first and last sample

# %%
import numpy as np
from matplotlib import pyplot as plt
import librosa
from ipywidgets import interact, fixed, FloatSlider
%matplotlib inline

Fs = 256
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


def pad_and_plot(t, x, Fs, pad_len_sec, pad_mode):
    pad_len = int(pad_len_sec * Fs)

    t = np.concatenate((np.arange(-pad_len, 0) / Fs, t,
                        np.arange(len(x), len(x) + pad_len) / Fs))
    x = np.pad(x, pad_len, pad_mode)
    N = len(x)

    plt.figure(figsize=(8, 1.5))
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(t, x, c='k')
    plt.xlim([-1.0, 11.0])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')

    ax2 = plt.subplot(1, 2, 2)
    X = np.abs(np.fft.fft(x)) / Fs
    freq = np.fft.fftfreq(N, d=1 / Fs)
    X = X[:N // 2]
    freq = freq[:N // 2]
    plt.plot(freq, X, c='k')
    plt.xlim([0, 7])
    plt.ylim([0, 3])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

    return ax1, ax2


print('No padding:')
ax1, ax2 = pad_and_plot(t, x, Fs, 0.0, 'constant')

print('Zero padding:')
ax1, ax2 = pad_and_plot(t, x, Fs, 1.0, 'constant')

print('Reflect padding:')
ax1, ax2 = pad_and_plot(t, x, Fs, 1.0, 'reflect')

# %% [markdown]
# ## Edge Phenomena
#
# Using different padding strategies, the beginning and end of the corresponding
# spectrograms will be different. When using a hop size smaller than N/2, more
# than one frame will be affected at both beginning and end.

# %%
def compute_stft(x, Fs, N, H, pad_mode='constant', center=True, color='gray_r'):
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N,
                     window='hann', pad_mode=pad_mode, center=center)
    Y = np.abs(X) ** 2
    Y = Y / np.max(Y)
    return Y


def plot_stft(Y, Fs, N, H, time_offset=0, time_unit='frames', xlim=None, ylim=None,
              title='', xlabel='', color='hot'):
    time_samples = np.arange(Y.shape[1])
    if time_unit == 'sec':
        time_sec = np.arange(Y.shape[1]) * (H / Fs) + time_offset
        extent = [time_sec[0] - H / (2 * Fs), time_sec[-1] + H / (2 * Fs), 0, Fs / 2]
        xlabel = 'Time (seconds)'
    else:
        time_samples = np.arange(Y.shape[1])
        extent = [time_samples[0] - 1 / 2, time_samples[-1] + 1 / 2, 0, Fs / 2]
        xlabel = 'Time (frames)'
    plt.imshow(Y, cmap=color, aspect='auto', origin='lower', extent=extent)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.colorbar()


N = 512
H = 128
xlim_frame = [-2, 22]
xlim_sec = [-1, 11]
ylim_hz = [0, 8]

plt.figure(figsize=(10, 6))

# No padding
Y = compute_stft(x, Fs, N, H, pad_mode=None, center=False)
plt.subplot(3, 2, 1)
plot_stft(Y, Fs, N, H, xlim=xlim_frame, ylim=ylim_hz, title='No padding')

plt.subplot(3, 2, 2)
plot_stft(Y, Fs, N, H, time_offset=N / (2 * Fs), time_unit='sec',
          xlim=xlim_sec, ylim=ylim_hz, title='No padding')

# Zero padding
Y = compute_stft(x, Fs, N, H, pad_mode='constant', center=True)
plt.subplot(3, 2, 3)
plot_stft(Y, Fs, N, H, xlim=xlim_frame, ylim=ylim_hz, title='Zero padding')

plt.subplot(3, 2, 4)
plot_stft(Y, Fs, N, H, time_unit='sec', xlim=xlim_sec, ylim=ylim_hz, title='Zero padding')

# Reflect padding
Y = compute_stft(x, Fs, N, H, pad_mode='reflect', center=True)
plt.subplot(3, 2, 5)
plot_stft(Y, Fs, N, H, xlim=xlim_frame, ylim=ylim_hz, title='Reflect padding')

plt.subplot(3, 2, 6)
time_sec = np.arange(Y.shape[1]) * (H / Fs)
plot_stft(Y, Fs, N, H, time_unit='sec', xlim=xlim_sec, ylim=ylim_hz, title='Reflect padding')

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller and Frank Zalkow.

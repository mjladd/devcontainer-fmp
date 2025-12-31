# %% [markdown]
# # STFT: Influence of Window Function
#
# The STFT depends on both the signal as well as the window function. The design
# of suitable window functions and their influence is a science by itself.
# Following Section 2.5 of [Muller, FMP, Springer 2015], we discuss some examples
# that illustrate how the window may affect the spectral estimate computed by the STFT.

# %% [markdown]
# ## Window Type
#
# The simplest way to obtain a local view is with a **rectangular window**, but it
# leads to discontinuities at the section's boundaries. These abrupt changes cause
# artifacts spread over the entire frequency spectrum.
#
# To attenuate boundary effects, one uses windows that are nonnegative and
# continuously fall to zero at the boundaries:
# - **Triangular window**: Smaller ripple artifacts
# - **Hann window**: Raised cosine, drops smoothly to zero, but introduces frequency smearing

# %%
import numpy as np
import scipy
from matplotlib import pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd

%matplotlib inline

duration = 2.0
Fs = 2000
omega = 10
N = int(duration * Fs)
t = np.arange(N) / Fs
x = 0.9 * np.sin(2 * np.pi * omega * t * t)

plt.figure(figsize=(8, 2))

plt.subplot(1, 2, 1)
plt.plot(t, x, c='k')
plt.xlim([t[0], t[-1]])
plt.ylim([-1.1, 1.1])
plt.xlabel('Time (seconds)')

plt.subplot(1, 2, 2)
X = np.abs(np.fft.fft(x)) / N * 2
freq = np.fft.fftfreq(N, d=1 / Fs)
X = X[:N // 2]
freq = freq[:N // 2]
plt.plot(freq, X, c='k')
plt.xlim([0, 50])
plt.ylim(bottom=0)
plt.xlabel('Frequency (Hz)')
plt.tight_layout()
plt.show()

# %%
def windowed_ft(t, x, Fs, w_pos_sec, w_len, w_type, upper_y=1.0):
    N = len(x)
    w_pos = int(Fs * w_pos_sec)
    w = np.zeros(N)
    w[w_pos:w_pos + w_len] = scipy.signal.get_window(w_type, w_len)
    x = x * w

    plt.figure(figsize=(8, 2))

    plt.subplot(1, 2, 1)
    plt.plot(t, x, c='k')
    plt.plot(t, w, c='r')
    plt.xlim([min(t), max(t)])
    plt.ylim([-1.1, 1.1])
    plt.xlabel('Time (seconds)')

    plt.subplot(1, 2, 2)
    X = np.abs(np.fft.fft(x)) / N * 2
    freq = np.fft.fftfreq(N, d=1 / Fs)
    X = X[:N // 2]
    freq = freq[:N // 2]
    plt.plot(freq, X, c='k')
    plt.xlim([0, 50])
    plt.ylim([0, upper_y])
    plt.xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()


w_len = 1024
w_pos = 1280
print('Rectangular window:')
windowed_ft(t, x, Fs, 1.0, w_len, 'boxcar', upper_y=0.15)
print('Triangular window:')
windowed_ft(t, x, Fs, 1.0, w_len, 'triang', upper_y=0.15)
print('Hann window:')
windowed_ft(t, x, Fs, 1.0, w_len, 'hann', upper_y=0.15)

# %% [markdown]
# ## Spectrogram: Effect of Window Type
#
# A chirp signal that linearly raises from 0 Hz to 400 Hz over 1 second.

# %%
duration = 1.0
Fs = 4000
N = int(duration * Fs)
t = np.arange(0, N) / Fs
x = np.sin(np.pi * 400 * t * t)

size_fade = 256
w_fade = np.hanning(size_fade * 2)[size_fade:]
x[-size_fade:] *= w_fade

ipd.display(ipd.Audio(x, rate=Fs))

plt.figure(figsize=(6.5, 1.5))
plt.plot(t, x, c='gray')
plt.xlabel('Time (seconds)')
plt.xlim([t[0], t[-1]])
plt.tight_layout()
plt.show()

# %% [markdown]
# The windows introduce some smearing of frequencies as well as additional
# ripple artifacts (weaker diagonal stripes). The ripple artifacts are stronger
# when using a rectangular window instead of a Hann window.

# %%
w_len_ms = 62.5
N = int((w_len_ms / 1000) * Fs)
H = 4
X_hann = librosa.stft(x, n_fft=N * 16, hop_length=H, win_length=N,
                      window='hann', center=True, pad_mode='constant')
X_rect = librosa.stft(x, n_fft=N * 16, hop_length=H, win_length=N,
                      window='boxcar', center=True, pad_mode='constant')

plt.figure(figsize=(8, 3))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(X_hann), ref=np.max),
                         y_axis='linear', x_axis='time', sr=Fs, hop_length=H, cmap='gray_r')
plt.clim([-80, 0])
plt.ylim([0, 500])
plt.xlim([0, 1])
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Hann window')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 3))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(X_rect), ref=np.max),
                         y_axis='linear', x_axis='time', sr=Fs, hop_length=H, cmap='gray_r')
plt.clim([-80, 0])
plt.ylim([0, 500])
plt.xlim([0, 1])
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Rectangular window')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Spectrogram: Effect of Window Size
#
# A signal consisting of two sinusoids (400 Hz and 450 Hz) with two impulses
# at t=0.45 and t=0.5 seconds.
#
# - Short window: Good time resolution (impulses separated), poor frequency resolution
# - Long window: Good frequency resolution (sinusoids separated), poor time resolution

# %%
duration = 1.0
Fs = 4000
N = int(duration * Fs)
t = np.arange(N) / Fs
x = np.sin(2 * np.pi * 400 * t) + np.sin(2 * np.pi * 450 * t)
x[int(round(0.45 * Fs))] = 10
x[int(round(0.50 * Fs))] = 10

ipd.display(ipd.Audio(x, rate=Fs))

plt.figure(figsize=(6.5, 1.5))
plt.plot(t, x, c='gray')
plt.xlabel('Time (seconds)')
plt.xlim([t[0], t[-1]])
plt.tight_layout()
plt.show()

# %%
w_len_ms = 32
N = int((w_len_ms / 1000) * Fs)
H = 16
X_short = librosa.stft(x, n_fft=N * 16, hop_length=H, win_length=N,
                       window='hann', center=True, pad_mode='constant')

w_len_ms = 128
N = int((w_len_ms / 1000) * Fs)
H = 16
X_long = librosa.stft(x, n_fft=N * 16, hop_length=H, win_length=N,
                      window='hann', center=True, pad_mode='constant')

plt.figure(figsize=(8, 3))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(X_short), ref=np.max),
                         y_axis='linear', x_axis='time', sr=Fs, hop_length=H, cmap='gray_r')
plt.clim([-90, 0])
plt.ylim([0, 1000])
plt.xlim([0, 1])
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Short Hann window')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 3))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(X_long), ref=np.max),
                         y_axis='linear', x_axis='time', sr=Fs, hop_length=H, cmap='gray_r')
plt.clim([-90, 0])
plt.ylim([0, 1000])
plt.xlim([0, 1])
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Long Hann window')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller and Frank Zalkow.

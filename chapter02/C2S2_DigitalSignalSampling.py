# %% [markdown]
# # Digital Signals: Sampling
#
# Analog signals have a continuous range of values in both time and amplitude.
# Since a computer can only store and process a finite number of values, one has
# to convert the waveform into some discrete representation - a process called
# **digitization**. The most common approach consists of two steps: **sampling**
# and **quantization**. This notebook explains sampling.

# %% [markdown]
# ## Sampling
#
# **Sampling** transforms a continuous-time (CT) signal into a discrete-time (DT)
# signal, which is defined only on a discrete subset of the time axis.
#
# **Equidistant sampling**: Given a positive real number T > 0, the DT-signal x
# is obtained by: x(n) := f(n * T)
#
# - x(n) is the **sample** taken at time t = n * T
# - T is the **sampling period**
# - Fs := 1/T is the **sampling rate** (in Hz)

# %%
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
%matplotlib inline


def generate_function(Fs, dur=1):
    """Generate example function"""
    N = int(Fs * dur)
    t = np.arange(N) / Fs
    x = 1 * np.sin(2 * np.pi * (2 * t - 0))
    x += 0.5 * np.sin(2 * np.pi * (6 * t - 0.1))
    x += 0.1 * np.sin(2 * np.pi * (20 * t - 0.2))
    return x, t


def sampling_equidistant(x_1, t_1, Fs_2, dur=None):
    """Equidistant sampling of interpolated signal"""
    if dur is None:
        dur = len(t_1) * t_1[1]
    N = int(Fs_2 * dur)
    t_2 = np.arange(N) / Fs_2
    x_2 = interp1d(t_1, x_1, kind='linear', fill_value='extrapolate')(t_2)
    return x_2, t_2


Fs_1 = 100
x_1, t_1 = generate_function(Fs=Fs_1, dur=2)

Fs_2 = 20
x_2, t_2 = sampling_equidistant(x_1, t_1, Fs_2)

plt.figure(figsize=(8, 2.2))
plt.plot(t_1, x_1, 'k')
plt.title('Original CT-signal')
plt.xlabel('Time (seconds)')
plt.ylim([-1.5, 1.5])
plt.xlim([t_1[0], t_1[-1]])
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 2.2))
plt.stem(t_2, x_2, linefmt='r', markerfmt='ro', basefmt='None')
plt.plot(t_1, x_1, 'k', linewidth=1, linestyle='dotted')
plt.title(r'Sampling rate $F_\mathrm{s} = %.0f$' % Fs_2)
plt.xlabel('Time (seconds)')
plt.ylim([-1.5, 1.5])
plt.xlim([t_1[0], t_1[-1]])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Aliasing
#
# Sampling is generally a **lossy** operation. Only if the CT-signal is
# **bandlimited**, a perfect reconstruction is possible (sampling theorem).
#
# Without additional properties, sampling may cause **aliasing** where certain
# frequency components become indistinguishable. When decreasing the sampling
# rate, higher-frequency components are not captured well.

# %%
def reconstruction_sinc(x, t, t_sinc):
    """Reconstruction from sampled signal using sinc-functions"""
    Fs = 1 / t[1]
    x_sinc = np.zeros(len(t_sinc))
    for n in range(0, len(t)):
        x_sinc += x[n] * np.sinc(Fs * t_sinc - n)
    return x_sinc


def plot_signal_reconstructed(t_1, x_1, t_2, x_2, t_sinc, x_sinc):
    plt.figure(figsize=(8, 2.2))
    plt.plot(t_1, x_1, 'k', linewidth=1, linestyle='dotted', label='Original signal')
    plt.stem(t_2, x_2, linefmt='r:', markerfmt='r.', basefmt='None', label='Samples')
    plt.plot(t_1, x_sinc, 'b', label='Reconstructed signal')
    plt.title(r'Sampling rate $F_\mathrm{s} = %.0f$' % (1 / t_2[1]))
    plt.xlabel('Time (seconds)')
    plt.ylim([-1.5, 1.5])
    plt.xlim([t_1[0], t_1[-1]])
    plt.legend(loc='upper right', framealpha=1)
    plt.tight_layout()
    plt.show()


Fs_2 = 40
x_2, t_2 = sampling_equidistant(x_1, t_1, Fs_2)
t_sinc = t_1
x_sinc = reconstruction_sinc(x_2, t_2, t_sinc)
plot_signal_reconstructed(t_1, x_1, t_2, x_2, t_sinc, x_sinc)

Fs_2 = 20
x_2, t_2 = sampling_equidistant(x_1, t_1, Fs_2)
t_sinc = t_1
x_sinc = reconstruction_sinc(x_2, t_2, t_sinc)
plot_signal_reconstructed(t_1, x_1, t_2, x_2, t_sinc, x_sinc)

Fs_2 = 10
x_2, t_2 = sampling_equidistant(x_1, t_1, Fs_2)
t_sinc = t_1
x_sinc = reconstruction_sinc(x_2, t_2, t_sinc)
plot_signal_reconstructed(t_1, x_1, t_2, x_2, t_sinc, x_sinc)

# %% [markdown]
# Aliasing also affects sound quality. Starting with a music signal at high
# sampling rate, successive reduction by factor of two.

# %%
import os
import IPython.display as ipd
import librosa
import scipy.signal

path_filename_wav = os.path.join('..', 'data', 'C2', 'FMP_C2_Sampling_C-major-scale.wav')
x, Fs = librosa.load(path_filename_wav, sr=8000)
Fs_orig = Fs
len_orig = len(x)
for i in range(5):
    print('Sampling rate Fs = %s; Number of samples = %s' % (Fs, len(x)), flush=True)
    # Some web browsers do not support arbitrary sample rates.
    # Work around: resample for playback
    x_play = scipy.signal.resample(x, len_orig)
    ipd.display(ipd.Audio(data=x_play, rate=Fs_orig))
    Fs = Fs // 2
    x = x[::2]

# %% [markdown]
# ## Sampling Theorem
#
# The **sampling theorem** (Nyquist-Shannon) states that a continuous-time signal
# that is bandlimited can be reconstructed perfectly under certain conditions.
#
# A CT-signal f is **Omega-bandlimited** if the Fourier transform vanishes for
# |omega| > Omega.
#
# Let f be Omega-bandlimited and x be its T-sampled version with T = 1/(2*Omega).
# Then f can be reconstructed from x by:
#
# f(t) = sum_{n} x(n) * sinc((t - n*T) / T)
#
# where sinc(t) = sin(pi*t) / (pi*t) for t != 0, and sinc(0) = 1.
#
# In other words, the CT-signal can be perfectly reconstructed if the bandlimit
# is no greater than half the sampling rate.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller.

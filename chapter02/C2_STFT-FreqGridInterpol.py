# %% [markdown]
# # STFT: Frequency Interpolation
#
# As an alternative to zero-padding for increasing frequency grid density,
# we discuss in this notebook interpolation techniques along the frequency
# direction to adjust the frequency resolution.

# %% [markdown]
# ## Interpolation
#
# Given a sequence of data points, the goal of **interpolation** is to compute
# intermediate data points that refine the sequence in some meaningful way.
#
# - **Piecewise constant (nearest-neighbor) interpolation**: For parameter t,
#   take the nearest t_n and define f*(t) = f(t_n). Typically yields a discontinuous function.
# - **Linear interpolation**: Gives a continuous function by connecting points with lines.
# - **Cubic interpolation**: Spline of third order for smoother results.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import librosa
%matplotlib inline

# Simulates the original function
t = np.arange(-0.5, 10.5, 0.01)
f = np.sin(2 * t)

# Known function values
t_n = np.arange(0, 11)
f_n = np.sin(2 * t_n)

# Piecewise constant interpolation
f_interpol_nearest = interp1d(t_n, f_n, kind='nearest', fill_value='extrapolate')(t)

plt.figure(figsize=(8, 2))
plt.plot(t, f, color=(0.8, 0.8, 0.8))
plt.plot(t, f_interpol_nearest, 'r-')
plt.plot(t_n, f_n, 'ko')
plt.ylim([-1.5, 1.5])
plt.title('Piecewise constant interpolation')
plt.tight_layout()
plt.show()

# %%
# Linear interpolation
f_interpol_linear = interp1d(t_n, f_n, kind='linear', fill_value='extrapolate')(t)

plt.figure(figsize=(8, 2))
plt.plot(t, f, color=(0.8, 0.8, 0.8))
plt.plot(t, f_interpol_linear, 'r-')
plt.plot(t_n, f_n, 'ko')
plt.ylim([-1.5, 1.5])
plt.title('Linear interpolation')
plt.tight_layout()
plt.show()

# %%
# Cubic interpolation
f_interpol_cubic = interp1d(t_n, f_n, kind='cubic', fill_value='extrapolate')(t)

plt.figure(figsize=(8, 2))
plt.plot(t, f, color=(0.8, 0.8, 0.8))
plt.plot(t, f_interpol_cubic, 'r-')
plt.plot(t_n, f_n, 'ko')
plt.ylim([-1.5, 1.5])
plt.title('Cubic interpolation')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Frequency Interpolation
#
# To increase the density of the linear frequency grid introduced by the DFT,
# we can apply interpolation techniques in the frequency domain. The index k
# of Y(k) corresponds to the physical frequency F_coef(k) = k * Fs / N.
#
# Introducing a factor rho, we refine the frequency grid to resolution Fs / (rho * N).

# %%
Fs = 32
duration = 2
omega1 = 5
omega2 = 15
N = int(duration * Fs)
t = np.arange(N) / Fs
t1 = t[:N // 2]
t2 = t[N // 2:]

x1 = 1.0 * np.sin(2 * np.pi * omega1 * t1)
x2 = 0.7 * np.sin(2 * np.pi * omega2 * t2)
x = np.concatenate((x1, x2))

plt.figure(figsize=(6, 2))
plt.plot(t, x, c='k')
plt.title('Original signal ($N$=%d)' % N)
plt.xlabel('Time (seconds)')
plt.xlim([t[0], t[-1]])
plt.tight_layout()
plt.show()

Y = np.abs(np.fft.fft(x)) / Fs
Y = Y[:N // 2 + 1]
F_coef = np.arange(N // 2 + 1) * Fs / N
plt.figure(figsize=(6, 2))
plt.plot(F_coef, Y, c='k')
plt.title('Magnitude DFT ($N$=%d)' % N)
plt.xlabel('Frequency (Hz)')
plt.xlim([F_coef[0], F_coef[-1]])
plt.tight_layout()
plt.show()


def interpolate_plot_DFT(N, Fs, F_coef, rho, int_method):
    F_coef_interpol = np.arange(F_coef[0], F_coef[-1], Fs / (rho * N))
    Y_interpol = interp1d(F_coef, Y, kind=int_method)(F_coef_interpol)
    plt.figure(figsize=(6, 2))
    plt.plot(F_coef_interpol, Y_interpol, c='k')
    plt.title(r'Magnitude DFT (interpolation: %s, $\rho$=%d)' % (int_method, rho))
    plt.xlabel('Frequency (Hz)')
    plt.xlim([F_coef[0], F_coef[-1]])
    plt.tight_layout()
    plt.show()


rho = 4
interpolate_plot_DFT(N=N, Fs=Fs, F_coef=F_coef, rho=rho, int_method='nearest')
interpolate_plot_DFT(N=N, Fs=Fs, F_coef=F_coef, rho=rho, int_method='linear')
interpolate_plot_DFT(N=N, Fs=Fs, F_coef=F_coef, rho=rho, int_method='cubic')

# %% [markdown]
# ## Frequency Interpolation for STFT
#
# To refine the frequency grid of an STFT, apply interpolation along the
# frequency direction.

# %%
import os
import IPython.display as ipd

# Load wav
fn_wav = os.path.join('..', 'data', 'C2', 'FMP_C2_F05c_C4_violin.wav')

Fs = 11025
x, Fs = librosa.load(fn_wav, sr=Fs)
ipd.display(ipd.Audio(x, rate=Fs))

t_wav = np.arange(0, x.shape[0]) * 1 / Fs
plt.figure(figsize=(6, 1))
plt.plot(t_wav, x, c='gray')
plt.xlim([t_wav[0], t_wav[-1]])
plt.xlabel('Time (seconds)')
plt.show()

# %%
def stft_convention_fmp(x, Fs, N, H, pad_mode='constant', center=True, mag=False, gamma=0):
    """Compute the discrete short-time Fourier transform (STFT)"""
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N,
                     window='hann', pad_mode=pad_mode, center=center)
    if mag:
        X = np.abs(X) ** 2
        if gamma > 0:
            X = np.log(1 + gamma * X)
    F_coef = librosa.fft_frequencies(sr=Fs, n_fft=N)
    T_coef = librosa.frames_to_time(np.arange(X.shape[1]), sr=Fs, hop_length=H)
    return X, T_coef, F_coef


def compute_f_coef_linear(N, Fs, rho=1):
    """Refines the frequency vector by factor of rho"""
    L = rho * N
    F_coef_new = np.arange(0, L // 2 + 1) * Fs / L
    return F_coef_new


def interpolate_freq_stft(Y, F_coef, F_coef_new):
    """Interpolation of STFT along frequency axis"""
    compute_Y_interpol = interp1d(F_coef, Y, kind='cubic', axis=0)
    Y_interpol = compute_Y_interpol(F_coef_new)
    return Y_interpol


def plot_compute_spectrogram_physical(x, Fs, N, H, xlim, ylim, rho=1, color='gray_r'):
    Y, T_coef, F_coef = stft_convention_fmp(x, Fs, N, H, mag=True, gamma=100)
    F_coef_new = compute_f_coef_linear(N, Fs, rho=rho)
    Y_interpol = interpolate_freq_stft(Y, F_coef, F_coef_new)
    extent = [T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]]
    plt.imshow(Y_interpol, cmap=color, aspect='auto', origin='lower', extent=extent)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.title(r'$\rho$=%d' % rho)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.colorbar()


xlim_sec = [2, 3]
ylim_hz = [2000, 3000]

N = 256
H = 64
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plot_compute_spectrogram_physical(x, Fs, N, H, xlim=xlim_sec, ylim=ylim_hz, rho=1)

plt.subplot(1, 3, 2)
plot_compute_spectrogram_physical(x, Fs, N, H, xlim=xlim_sec, ylim=ylim_hz, rho=2)

plt.subplot(1, 3, 3)
plot_compute_spectrogram_physical(x, Fs, N, H, xlim=xlim_sec, ylim=ylim_hz, rho=4)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Log-Frequency STFT via Interpolation
#
# Interpolation can convert the linearly spaced frequency axis (Hz) into a
# logarithmically spaced axis (pitches or cents). This results in a
# **log-frequency spectrogram**.
#
# **Cents** is a logarithmic unit for musical intervals. Given a reference
# frequency omega_0, the distance from omega to omega_0 is:
# log2(omega / omega_0) * 1200 cents

# %%
def compute_f_coef_log(R, F_min, F_max):
    """Adapts the frequency vector in a logarithmic fashion

    Args:
        R (scalar): Resolution (cents)
        F_min (float): Minimum frequency
        F_max (float): Maximum frequency (not included)

    Returns:
        F_coef_log (np.ndarray): Refined frequency vector (Hz)
        F_coef_cents (np.ndarray): Refined frequency vector (cents)
    """
    n_bins = np.ceil(1200 * np.log2(F_max / F_min) / R).astype(int)
    F_coef_log = 2 ** (np.arange(0, n_bins) * R / 1200) * F_min
    F_coef_cents = 1200 * np.log2(F_coef_log / F_min)
    return F_coef_log, F_coef_cents


N = 1024
H = 256
Y, T_coef, F_coef = stft_convention_fmp(x, Fs, N, H, mag=True, gamma=100)

F_min = 100
F_max = 3200
R = 20
F_coef_log, F_coef_cents = compute_f_coef_log(R, F_min, F_max)

print('#bins=%3d, F_coef[0]      =%6.2f, F_coef[1]      =%6.2f, F_coef[-1]      =%6.2f' %
      (len(F_coef), F_coef[0], F_coef[1], F_coef[-1]))
print('#bins=%3d, F_coef_log[0]  =%6.2f, F_coef_log[1]  =%6.2f, F_coef_log[-1]  =%6.2f' %
      (len(F_coef_log), F_coef_log[0], F_coef_log[1], F_coef_log[-1]))
print('#bins=%3d, F_coef_cents[0]=%6.2f, F_coef_cents[1]=%6.2f, F_coef_cents[-1]=%6.2f' %
      (len(F_coef_cents), F_coef_cents[0], F_coef_cents[1], F_coef_cents[-1]))

# %%
Y_interpol = interpolate_freq_stft(Y, F_coef, F_coef_log)
color = 'gray_r'

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
extent = [T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]]
plt.imshow(Y, cmap=color, aspect='auto', origin='lower', extent=extent)
y_ticks_freq = np.array([100, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200])
plt.yticks(y_ticks_freq)
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Linear frequency axis')
plt.ylim([F_min, F_max])
plt.colorbar()

plt.subplot(1, 3, 2)
extent = [T_coef[0], T_coef[-1], F_coef_cents[0], F_coef_cents[-1]]
plt.imshow(Y_interpol, cmap=color, aspect='auto', origin='lower', extent=extent)
y_tick_freq_cents = 1200 * np.log2(y_ticks_freq / F_min)
plt.yticks(y_tick_freq_cents, y_ticks_freq)
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Log-frequency axis with R=%d' % R)
plt.colorbar()

plt.subplot(1, 3, 3)
extent = [T_coef[0], T_coef[-1], F_coef_cents[0], F_coef_cents[-1]]
plt.imshow(Y_interpol, cmap=color, aspect='auto', origin='lower', extent=extent)
y_ticks_cents = np.array([0, 1200, 2400, 3600, 4800, 6000])
plt.yticks(y_ticks_cents)
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (cents)')
plt.title('Log-frequency axis with R=%d' % R)
plt.colorbar()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller and Sebastian Rosenzweig.

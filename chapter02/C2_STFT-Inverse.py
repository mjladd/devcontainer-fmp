# %% [markdown]
# # STFT: Inverse
#
# In this notebook, we introduce the inverse DFT and then show how a signal can
# be recovered from its STFT under relatively mild conditions on the windowing process.

# %% [markdown]
# ## Inverse DFT
#
# Given a vector x in C^N of length N, the discrete Fourier transform is defined by:
# X = DFT_N * x
#
# The DFT is invertible. The inverse DFT is:
# x = DFT_N^{-1} * X
#
# where DFT_N^{-1}(n, k) = (1/N) * exp(2*pi*i*k*n/N)

# %%
import os
import numpy as np
import scipy.signal
import librosa
from numba import jit
from matplotlib import pyplot as plt

import sys
sys.path.append('..')
import libfmp.c2

%matplotlib inline


@jit(nopython=True)
def generate_matrix_dft_inv(N, K):
    """Generates an IDFT (inverse discrete Fourier transform) matrix

    Args:
        N (int): Number of samples
        K (int): Number of frequency bins

    Returns:
        dft (np.ndarray): The IDFT matrix
    """
    dft = np.zeros((K, N), dtype=np.complex128)
    for n in range(N):
        for k in range(K):
            dft[k, n] = np.exp(2j * np.pi * k * n / N) / N
    return dft


N = 32
dft_mat = libfmp.c2.generate_matrix_dft(N, N)
dft_mat_inv = generate_matrix_dft_inv(N, N)

I = np.eye(N)
A = np.dot(dft_mat, dft_mat_inv)
B = np.dot(dft_mat_inv, dft_mat)

plt.figure(figsize=(11, 3))

plt.subplot(1, 3, 1)
plt.title(r'$I_N$ for $N = %d$' % N)
plt.imshow(I, origin='lower', cmap='seismic', aspect='equal')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title(r'$|I_N - \mathrm{DFT}_N \cdot \mathrm{DFT}_N^{-1}|$')
plt.imshow(np.abs(I - A), origin='lower', cmap='seismic', aspect='equal')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title(r'$|I_N - \mathrm{DFT}_N^{-1} \cdot \mathrm{DFT}_N|$')
plt.imshow(np.abs(I - B), origin='lower', cmap='seismic', aspect='equal')
plt.colorbar()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Inverse STFT
#
# Let x_n(r) := x(r+nH)*w(r) be the windowed signal. The STFT coefficients X(n,k)
# for k in [0:N-1] are obtained via DFT. Since DFT_N is invertible, we can
# reconstruct the windowed signal x_n from the STFT.
#
# To obtain the samples x(r), we use the **overlap-add technique**:
# x(r) = sum_{n} x_n(r-nH) / sum_{n} w(r-nH)
#
# as long as sum_{n} w(r-nH) != 0 for all r.

# %% [markdown]
# ## Partition of Unity
#
# Often, one chooses a window function and a hop size such that:
# sum_{n} w(r-nH) = 1 for all r.
#
# In this case, the time-shifted window functions define a **partition of unity**.
# For example, using the squared sinusoidal window w(r) = sin(pi*r/N)^2
# with hop size H = N/2 yields a partition of unity.

# %%
def plot_sum_window(w, H, L, title='', figsize=(5, 1.5)):
    N = len(w)
    M = np.floor((L - N) / H).astype(int) + 1
    w_sum = np.zeros(L)
    plt.figure(figsize=figsize)
    for m in range(M):
        w_shifted = np.zeros(L)
        w_shifted[m * H:m * H + N] = w
        plt.plot(w_shifted, 'k')
        w_sum = w_sum + w_shifted
    plt.plot(w_sum, 'r', linewidth=3)
    plt.xlim([0, L - 1])
    plt.ylim([0, 1.1 * np.max(w_sum)])
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return w_sum


L = 256
N = 64

H = N // 2
w_type = 'triang'
w = scipy.signal.get_window(w_type, N)
plot_sum_window(w, H, L, title='Triangular window, H = N/2')

H = N // 2
w_type = 'hann'
w = scipy.signal.get_window(w_type, N)
plot_sum_window(w, H, L, title='Hann window, H = N/2')

H = 3 * N // 8
w_type = 'hann'
w = scipy.signal.get_window(w_type, N)
plot_sum_window(w, H, L, title='Hann window, H = 3N/8')

H = N // 4
w = scipy.signal.windows.gaussian(N, std=8)
plot_sum_window(w, H, L, title='Gaussian window, H = N/4')

# %% [markdown]
# ## Basic Implementation of STFT and Inverse STFT

# %%
def istft_basic(X, w, H, L):
    """Compute the inverse of the basic discrete short-time Fourier transform (ISTFT)

    Args:
        X (np.ndarray): The discrete short-time Fourier transform
        w (np.ndarray): Window function
        H (int): Hopsize
        L (int): Length of time signal

    Returns:
        x (np.ndarray): Time signal
    """
    N = len(w)
    M = X.shape[1]
    x_win_sum = np.zeros(L)
    w_sum = np.zeros(L)
    for m in range(M):
        x_win = np.fft.ifft(X[:, m])
        # Avoid imaginary values (due to floating point arithmetic)
        x_win = np.real(x_win)
        x_win_sum[m * H:m * H + N] = x_win_sum[m * H:m * H + N] + x_win
        w_shifted = np.zeros(L)
        w_shifted[m * H:m * H + N] = w
        w_sum = w_sum + w_shifted
    # Avoid division by zero
    w_sum[w_sum == 0] = np.finfo(np.float32).eps
    x_rec = x_win_sum / w_sum
    return x_rec, x_win_sum, w_sum


L = 256
t = np.arange(L) / L
omega = 4
x = np.sin(2 * np.pi * omega * t * t)

N = 64
H = 3 * N // 8
w_type = 'hann'
w = scipy.signal.get_window(w_type, N)
X = libfmp.c2.stft_basic(x, w=w, H=H)
x_rec, x_win_sum, w_sum = istft_basic(X, w=w, H=H, L=L)

plt.figure(figsize=(8, 3))
plt.plot(x, color=[0, 0, 0], linewidth=4, label='Original signal')
plt.plot(x_win_sum, 'b', label='Summed windowed signals')
plt.plot(w_sum, 'r', label='Summed windows')
plt.plot(x_rec, color=[0.8, 0.8, 0.8], linestyle=':', linewidth=4, label='Reconstructed signal')
plt.xlim([0, L - 1])
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## LibROSA Implementation
#
# The librosa package offers `librosa.stft` and `librosa.istft` for computing
# the STFT and its inverse. It's important to compensate for possible padding.

# %%
def print_plot(x, x_rec):
    print('Number of samples of x:    ', x.shape[0])
    print('Number of samples of x_rec:', x_rec.shape[0])
    if x.shape[0] == x_rec.shape[0]:
        print('Signals x and x_inv agree:', np.allclose(x, x_rec))
        plt.figure(figsize=(6, 2))
        plt.plot(x - x_rec, color='red')
        plt.xlim([0, x.shape[0]])
        plt.title('Differences between x and x_rec')
        plt.xlabel('Time (samples)')
        plt.tight_layout()
        plt.show()
    else:
        print('Number of samples of x and x_rec does not agree.')


fn_wav = os.path.join('..', 'data', 'C2', 'FMP_C2_F05c_C4_violin.wav')
Fs = 11025
x, Fs = librosa.load(fn_wav, sr=Fs)

N = 4096
H = 2048
L = x.shape[0]

print('=== Centered Case ===')
X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann',
                 pad_mode='constant', center=True)
x_rec = librosa.istft(X, hop_length=H, win_length=N, window='hann', center=True, length=L)
print('stft: center=True; istft: center=True')
print_plot(x, x_rec)

print('=== Non-Centered Case ===')
X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann',
                 pad_mode='constant', center=False)
x_rec = librosa.istft(X, hop_length=H, win_length=N, window='hann', center=False, length=L)
print('stft: center=False; istft: center=False')
print_plot(x, x_rec)

# %% [markdown]
# ## libfmp Implementation

# %%
from libfmp.c2 import fft, stft, ifft, istft

L = 256
t = np.arange(L) / L
omega = 4
x = np.sin(2 * np.pi * omega * t * t)

X = fft(x)
x_rec = np.real(ifft(X))
plt.figure(figsize=(8, 2))
plt.plot(x, color='k', linewidth=4, label='Original waveform')
plt.plot(x_rec, color=[0.8, 0.8, 0.8], linestyle=':', linewidth=4, label='Reconstructed with inverse FFT')
plt.xlim([0, L - 1])
plt.xlabel('Time index')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()

N = 64
H = 3 * N // 8
w = scipy.signal.get_window('hann', N)
X_stft = stft(x, w=w, H=H)
x_stft_rec = istft(X_stft, w=w, H=H, L=L)
plt.figure(figsize=(8, 2))
plt.plot(x, color='k', linewidth=4, label='Original waveform')
plt.plot(x_stft_rec, color=[0.8, 0.8, 0.8], linestyle=':', linewidth=4, label='Reconstructed with inverse STFT')
plt.xlim([0, L - 1])
plt.xlabel('Time index')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller and Frank Zalkow.

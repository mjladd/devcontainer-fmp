# %% [markdown]
# # Logarithmic Compression
#
# Following Section 3.1.2.1 of [Müller, FMP, Springer 2015], we introduce in this
# notebook the concept of logarithmic compression.

# %% [markdown]
# ## Compression Function
#
# In music signal processing, the problem with representations such as a spectrogram
# or chromagram is that its values possess a large dynamic range. One often uses a
# **decibel scale** to balance out this discrepancy. More generally, one may apply
# other types of logarithm-based functions, a step often referred to as **logarithmic
# compression**. Let gamma be a positive constant, then we define:
#
# Gamma_gamma(v) := log(1 + gamma * v)
#
# The degree of compression can be adjusted by the constant gamma: the larger gamma,
# the larger the resulting compression.

# %%
import os
import numpy as np
from matplotlib import pyplot as plt
import librosa
import librosa.display
from numba import jit
%matplotlib inline


@jit(nopython=True)
def log_compression(v, gamma=1.0):
    """Logarithmically compresses a value or array

    Args:
        v (float or np.ndarray): Value or array
        gamma (float): Compression factor

    Returns:
        v_compressed (float or np.ndarray): Compressed value or array
    """
    return np.log(1 + gamma * v)


v = np.arange(1001) / 100

plt.figure(figsize=(5, 5))
plt.plot(v, v, color='black', linestyle=':', label='Identity')
plt.plot(v, log_compression(v, gamma=1), color='blue', label='$\gamma$ = 1')
plt.plot(v, log_compression(v, gamma=10), color='gray', label='$\gamma$ = 10')
plt.plot(v, log_compression(v, gamma=100), color='red', label='$\gamma$ = 100')
plt.xlabel('Original values')
plt.ylabel('Compressed values')
plt.xlim([v[0], v[-1]])
plt.ylim([v[0], v[-1]])
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()

# %% [markdown]
# ## Compressed Spectrogram
#
# For a representation with positive values such as a spectrogram, one obtains a
# compressed version by applying the function Gamma_gamma to each of the values.

# %%
x, Fs = librosa.load(os.path.join('..', 'data', 'C3', 'FMP_C3_NoteC4_Piano.wav'))

N = 1024
H = 512
X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann', pad_mode='constant', center=True)
T_coef = np.arange(X.shape[1]) * H / Fs
K = N // 2
F_coef = np.arange(K + 1) * Fs / N
Y = np.abs(X) ** 2

plt.figure(figsize=(11, 3))
extent = [T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]]
gamma_set = [0, 1, 100, 10000]
M = len(gamma_set)
Y = np.abs(X)

for m in range(M):
    ax = plt.subplot(1, M, m + 1)
    gamma = gamma_set[m]
    if gamma == 0:
        Y_compressed = Y
        title = 'No compression'
    else:
        Y_compressed = log_compression(Y, gamma=gamma)
        title = '$\gamma$=%d' % gamma
    plt.imshow(Y_compressed, cmap='gray_r', aspect='auto', origin='lower', extent=extent)
    plt.xlabel('Time (seconds)')
    plt.ylim([0, 4000])
    plt.clim([0, Y_compressed.max()])
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.title(title)

plt.tight_layout()

# %% [markdown]
# While the partials (the horizontal lines) are hardly visible in the original
# spectrogram, they clearly pop up in the compressed versions. Also, the note
# transient (the vertical line at time position t=0.9) emerges when increasing
# the constant gamma.

# %% [markdown]
# ## Compressed Chromagram
#
# Next, we consider a chromagram and its compressed version.

# %%
Fs = 22050
fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3_F08_C-major-scale_pause.wav')
x, Fs = librosa.load(fn_wav, sr=Fs)

N = 4096
H = 512
C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
C = C / C.max()

plt.figure(figsize=(8, 8))
gamma_set = [0, 10, 1000, 100000]
M = len(gamma_set)
Y = np.abs(X)

for m in range(M):
    ax = plt.subplot(M, 1, m + 1)
    gamma = gamma_set[m]
    if gamma == 0:
        C_compressed = C
        title = 'No compression'
    else:
        C_compressed = log_compression(C, gamma=gamma)
        title = '$\gamma$=%d' % gamma
    librosa.display.specshow(C_compressed, x_axis='time',
                             y_axis='chroma', cmap='gray_r', sr=Fs, hop_length=H)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Chroma')
    plt.clim([0, np.max(C_compressed)])
    plt.title(title)
    plt.colorbar()

plt.tight_layout()

# %% [markdown]
# ## Further Notes
#
# - Logarithmic compression is a simple, yet powerful tool that is widely used for
#   various music processing tasks. We use this concept throughout the FMP notebooks.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

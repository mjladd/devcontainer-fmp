# %% [markdown]
# # Log-Frequency Spectrogram and Chromagram
#
# Following Section 3.1.1 of [Müller, FMP, Springer 2015], we introduce in this
# notebook two feature representations known as log-frequency spectrogram and chromagram.

# %% [markdown]
# ## STFT and Pitch Frequencies
#
# Assuming that we are dealing with music whose pitches can be meaningfully categorized
# according to the equal-tempered scale, we show how an audio recording can be transformed
# into a feature representation that reveals the distribution of the signal's energy
# across the different pitches. Such features can be obtained from a spectrogram by
# converting the linear frequency axis (measured in Hertz) into a logarithmic axis
# (measured in pitches). The resulting representation is also called **log-frequency
# spectrogram**.

# %%
import os
import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt
from numba import jit
import librosa
import pandas as pd
import IPython.display as ipd

import sys
sys.path.append('..')
import libfmp.c2

%matplotlib inline

# Load wav
fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3_F03.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, sr=Fs)

# Compute Magnitude STFT
N = 4096
H = 1024
X, T_coef, F_coef = libfmp.c2.stft_convention_fmp(x, Fs, N, H)
Y = np.abs(X) ** 2

# Plot spectrogram
fig = plt.figure(figsize=(10, 4))
eps = np.finfo(float).eps
plt.imshow(10 * np.log10(eps + Y), origin='lower', aspect='auto', cmap='gray_r',
           extent=[T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]])
plt.clim([-30, 30])
plt.ylim([0, 4500])
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
cbar = plt.colorbar()
cbar.set_label('Magnitude (dB)')
plt.tight_layout()

# Plot rectangle corresponding to pitch C3 (p=48)
rect = matplotlib.patches.Rectangle((29.3, 0.5), 1.2, 4490, linewidth=3,
                                    edgecolor='r', facecolor='none')
plt.gca().add_patch(rect)
plt.text(28, -400, r'$\mathrm{C3}$', color='r', fontsize='x-large');

# %% [markdown]
# ## Logarithmic Frequency Pooling
#
# The logarithmic perception of frequency motivates the use of a time-frequency
# representation with a logarithmic frequency axis labeled by the pitches of the
# equal-tempered scale. To derive such a representation from a given spectrogram
# representation, the basic idea is to assign each spectral coefficient X(n,k) to
# the pitch with a center frequency that is closest to the frequency F_coef(k).

# %%
def note_name(p):
    """Returns note name of pitch

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        p (int): Pitch value

    Returns:
        name (str): Note name
    """
    chroma = ['A', 'A$^\\sharp$', 'B', 'C', 'C$^\\sharp$', 'D', 'D$^\\sharp$', 'E', 'F', 'F$^\\sharp$', 'G',
              'G$^\\sharp$']
    name = chroma[(p - 69) % 12] + str(p // 12 - 1)
    return name

f_pitch = lambda p: 440 * 2 ** ((p - 69) / 12)

note_infos = []
for p in range(60, 73):
    name = note_name(p)
    p_pitch = f_pitch(p)
    p_pitch_lower = f_pitch(p - 0.5)
    p_pitch_upper = f_pitch(p + 0.5)
    bw = p_pitch_upper - p_pitch_lower
    note_infos.append([name, p, p_pitch, p_pitch_lower, p_pitch_upper, bw])

df = pd.DataFrame(note_infos, columns=['Note', '$p$',
                                       '$F_\\mathrm{pitch}(p)$',
                                       '$F_\\mathrm{pitch}(p-0.5)$',
                                       '$F_\\mathrm{pitch}(p+0.5)$',
                                       '$\\mathrm{BW}(p)$'])


html = df.to_html(index=False, float_format='%.2f')
html = html.replace('<table', '<table style="width: 80%"')
ipd.HTML(html)

# %% [markdown]
# ## Log-Frequency Spectrogram
#
# Based on the sets P(p), we obtain a log-frequency spectrogram Y_LF using a simple
# pooling procedure by summing up all spectral coefficients that belong to the same pitch.

# %%
@jit(nopython=True)
def f_pitch(p, pitch_ref=69, freq_ref=440.0):
    """Computes the center frequency/ies of a MIDI pitch

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        p (float): MIDI pitch value(s)
        pitch_ref (float): Reference pitch (default: 69)
        freq_ref (float): Frequency of reference pitch (default: 440.0)

    Returns:
        freqs (float): Frequency value(s)
    """
    return 2 ** ((p - pitch_ref) / 12) * freq_ref

@jit(nopython=True)
def pool_pitch(p, Fs, N, pitch_ref=69, freq_ref=440.0):
    """Computes the set of frequency indices that are assigned to a given pitch

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        p (float): MIDI pitch value
        Fs (scalar): Sampling rate
        N (int): Window size of Fourier fransform
        pitch_ref (float): Reference pitch (default: 69)
        freq_ref (float): Frequency of reference pitch (default: 440.0)

    Returns:
        k (np.ndarray): Set of frequency indices
    """
    lower = f_pitch(p - 0.5, pitch_ref, freq_ref)
    upper = f_pitch(p + 0.5, pitch_ref, freq_ref)
    k = np.arange(N // 2 + 1)
    k_freq = k * Fs / N  # F_coef(k, Fs, N)
    mask = np.logical_and(lower <= k_freq, k_freq < upper)
    return k[mask]

@jit(nopython=True)
def compute_spec_log_freq(Y, Fs, N):
    """Computes a log-frequency spectrogram

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        Y (np.ndarray): Magnitude or power spectrogram
        Fs (scalar): Sampling rate
        N (int): Window size of Fourier fransform

    Returns:
        Y_LF (np.ndarray): Log-frequency spectrogram
        F_coef_pitch (np.ndarray): Pitch values
    """
    Y_LF = np.zeros((128, Y.shape[1]))
    for p in range(128):
        k = pool_pitch(p, Fs, N)
        Y_LF[p, :] = Y[k, :].sum(axis=0)
    F_coef_pitch = np.arange(128)
    return Y_LF, F_coef_pitch

Y_LF, F_coef_pitch = compute_spec_log_freq(Y, Fs, N)

fig = plt.figure(figsize=(10, 4))
plt.imshow(10 * np.log10(eps + Y_LF), origin='lower', aspect='auto', cmap='gray_r',
           extent=[T_coef[0], T_coef[-1], 0, 127])
plt.clim([-10, 50])
plt.ylim([21, 108])
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (pitch)')
cbar = plt.colorbar()
cbar.set_label('Magnitude (dB)')

plt.tight_layout()

# Create a Rectangle patch
rect = matplotlib.patches.Rectangle((29.3, 21), 1.2, 86.5, linewidth=3, edgecolor='r', facecolor='none')
plt.gca().add_patch(rect)
plt.text(28, 15, r'$\mathrm{C3}$', color='r', fontsize='x-large');

# %% [markdown]
# Looking at the spectrogram visualization, one can make some interesting observations:
#
# * As a general trend, the sounds for higher notes possess a cleaner harmonic spectrum
#   than the ones for lower notes.
# * The vertical stripes indicate that some of the signal's energy is spread over large
#   parts of the spectrum due to inharmonicities, transient and resonance effects.
# * The frequency content depends on the microphone's frequency response.

# %%
print('Sampling rate: Fs = ', Fs)
print('Window size: N = ', N)
print('STFT frequency resolution (in Hz): Fs/N = %4.2f' % (Fs / N))

for p in [76, 64, 52, 40, 39, 38]:
    print('Set P(%d) = %s' % (p, pool_pitch(p, Fs, N)))

# %% [markdown]
# ## Chromagram
#
# The main idea of **chroma features** is to aggregate all spectral information that
# relates to a given pitch class into a single coefficient. Given a pitch-based
# log-frequency spectrogram, a **chromagram** can be derived by summing up all pitch
# coefficients that belong to the same chroma.

# %%
@jit(nopython=True)
def compute_chromagram(Y_LF):
    """Computes a chromagram

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        Y_LF (np.ndarray): Log-frequency spectrogram

    Returns:
        C (np.ndarray): Chromagram
    """
    C = np.zeros((12, Y_LF.shape[1]))
    p = np.arange(128)
    for c in range(12):
        mask = (p % 12) == c
        C[c, :] = Y_LF[mask, :].sum(axis=0)
    return C

C = compute_chromagram(Y_LF)

fig = plt.figure(figsize=(10, 3))
chroma_label = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
plt.imshow(10 * np.log10(eps + C), origin='lower', aspect='auto', cmap='gray_r',
           extent=[T_coef[0], T_coef[-1], 0, 12])
plt.clim([0, 60])
plt.xlabel('Time (seconds)')
plt.ylabel('Chroma')
cbar = plt.colorbar()
cbar.set_label('Magnitude (dB)')
plt.yticks(np.arange(12) + 0.5, chroma_label)
plt.tight_layout()

rect = matplotlib.patches.Rectangle((29.3, 0.0), 1.2, 12, linewidth=3, edgecolor='r', facecolor='none')
plt.gca().add_patch(rect)
plt.text(28.5, -1.2, r'$\mathrm{C3}$', color='r', fontsize='x-large');

# %% [markdown]
# ## Example: Burgmüller
#
# As an illustrating example, we now consider the first four measures of Op. 100, No. 2
# by Friedrich Burgmüller.

# %%
fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3_F05.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav)
# ipd.display(ipd.Audio(x, rate=Fs))

N = 4096
H = 512
X, T_coef, F_coef = libfmp.c2.stft_convention_fmp(x, Fs, N, H)
eps = np.finfo(float).eps
Y = np.abs(X) ** 2
Y_LF, F_coef_pitch = compute_spec_log_freq(Y, Fs, N)
C = compute_chromagram(Y_LF)

fig = plt.figure(figsize=(8,3))
plt.imshow(10 * np.log10(eps + Y_LF), origin='lower', aspect='auto', cmap='gray_r',
           extent=[T_coef[0], T_coef[-1], 0, 128])
plt.clim([-10, 50])
plt.ylim([55, 92])
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (pitch)')
cbar = plt.colorbar()
cbar.set_label('Magnitude (dB)')
plt.tight_layout()

fig = plt.figure(figsize=(8, 2.5))
plt.imshow(10 * np.log10(eps + C), origin='lower', aspect='auto', cmap='gray_r',
           extent=[T_coef[0], T_coef[-1], 0, 12])
plt.clim([0, 50])
plt.xlabel('Time (seconds)')
plt.ylabel('Chroma')
cbar = plt.colorbar()
cbar.set_label('Magnitude (dB)')
plt.yticks(np.arange(12) + 0.5, chroma_label)
plt.tight_layout()

# %% [markdown]
# ## libfmp Implementation
#
# The basic functions for computing a log-frequency spectrogram and a chromagram have
# been included into `libfmp`.

# %%
import libfmp.c3

fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3_F05.wav')
x, Fs = librosa.load(fn_wav)

N, H = 4096, 512
X, T_coef, F_coef = libfmp.c2.stft_convention_fmp(x, Fs, N, H)
Y = np.abs(X) ** 2
Y_LF, F_coef_pitch = libfmp.c3.compute_spec_log_freq(Y, Fs, N)
C = libfmp.c3.compute_chromagram(Y_LF)

fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.02],
                                          'height_ratios': [3, 2]}, figsize=(8, 5))

libfmp.b.plot_matrix(10 * np.log10(eps + Y_LF), Fs=Fs/H, ax=[ax[0,0], ax[0,1]],
        ylim=[55,92], clim=[0, 50], title='Log-frequency spectrogram',
        ylabel='Frequency (pitch)', colorbar=True, cbar_label='Magnitude (dB)');

libfmp.b.plot_chromagram(10 * np.log10(eps + C), Fs=Fs/H, ax=[ax[1,0], ax[1,1]],
        chroma_yticks = [0,4,7,11], clim=[10, 50], title='Chromagram',
        ylabel='Chroma', colorbar=True, cbar_label='Magnitude (dB)');

plt.tight_layout()

# %% [markdown]
# ## Further Notes
#
# * There are many variants for computing log-frequency spectrograms and chromagrams.
# * LibROSA also offers various functionalities for computing and visualizing
#   spectrograms, chromagrams, and other feature representations.

# %%
import librosa, librosa.display
C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
plt.figure(figsize=(8, 2))
librosa.display.specshow(10 * np.log10(eps + C), x_axis='time',
                         y_axis='chroma', sr=Fs, hop_length=H)
plt.colorbar();

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Frank Zalkow and Meinard Müller.

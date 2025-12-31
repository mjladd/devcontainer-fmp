# %% [markdown]
# # Dynamics, Intensity, and Loudness
#
# Following Section 1.3.3 of [Muller, FMP, Springer 2015], we introduce concepts
# related to dynamics, intensity, and loudness.

# %% [markdown]
# ## Decibel Scale
#
# **Sound power** expresses energy per unit time emitted by a sound source.
# **Sound intensity** is power per unit area.
#
# Key reference values:
# - Threshold of Hearing (TOH): 10^-12 W/m^2
# - Threshold of Pain (TOP): 10 W/m^2
#
# The **decibel (dB)** scale is logarithmic:
# dB(I) = 10 * log10(I / I_TOH)
#
# Properties:
# - dB(I_TOH) = 0
# - Doubling intensity = ~3 dB increase

# %%
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
import sys

sys.path.append('..')
import libfmp.b

%matplotlib inline


def compute_power_db(x, Fs, win_len_sec=0.1, power_ref=10 ** (-12)):
    """Compute signal power in dB

    Args:
        x (np.ndarray): Signal to analyze
        Fs (scalar): Sampling rate
        win_len_sec (float): Window length in seconds
        power_ref (float): Reference power level (0 dB)

    Returns:
        power_db (np.ndarray): Signal power in dB
    """
    win_len = round(win_len_sec * Fs)
    win = np.ones(win_len) / win_len
    power_db = 10 * np.log10(np.convolve(x ** 2, win, mode='same') / power_ref)
    return power_db


# %% [markdown]
# ## Beethoven Example
#
# Computing the sound power level over time for Beethoven's Fifth Symphony
# (Herbert von Karajan, 1946). The dynamic range is roughly 75 to 105 dB.

# %%
fn_wav = os.path.join('..', 'data', 'C1', 'FMP_C1_F10_Beethoven_Fifth-MM1-21_Karajan1946.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, sr=Fs, mono=True)

win_len_sec = 0.2
power_db = compute_power_db(x, win_len_sec=win_len_sec, Fs=Fs)

# Plot waveform
libfmp.b.plot_signal(x, Fs=Fs, ylabel='Amplitude')
plt.title('Waveform - Beethoven Fifth (Karajan 1946)')
plt.show()

# Plot power level
libfmp.b.plot_signal(power_db, Fs=Fs, ylabel='Power (dB)', color='red')
plt.ylim([70, 110])
plt.title('Sound Power Level (dB)')
plt.show()

# %% [markdown]
# ## Loudness
#
# **Loudness** is a subjective measure - the perceived volume of a sound.
# It depends on:
# - Individual listener (age, hearing ability)
# - Sound duration
# - Sound frequency
#
# Humans are most sensitive to frequencies around 2-4 kHz. The **phon** unit
# expresses perceived loudness of pure tones. **Equal loudness contours** show
# the intensity levels required for different frequencies to sound equally loud.

# %%
def compute_equal_loudness_contour(freq_min=30, freq_max=15000, num_points=100):
    """Compute an equal loudness contour approximation

    Args:
        freq_min (float): Lowest frequency
        freq_max (float): Highest frequency
        num_points (int): Number of evaluation points

    Returns:
        equal_loudness_contour (np.ndarray): Contour in dB
        freq_range (np.ndarray): Frequency points
    """
    freq_range = np.logspace(np.log10(freq_min), np.log10(freq_max), num=num_points)
    freq = 1000
    # D-weighting function approximation
    h_freq = ((1037918.48 - freq ** 2) ** 2 + 1080768.16 * freq ** 2) / (
            (9837328 - freq ** 2) ** 2 + 11723776 * freq ** 2)
    n_freq = (freq / (6.8966888496476 * 10 ** (-5))) * np.sqrt(
        h_freq / ((freq ** 2 + 79919.29) * (freq ** 2 + 1345600)))
    h_freq_range = ((1037918.48 - freq_range ** 2) ** 2 + 1080768.16 * freq_range ** 2) / (
            (9837328 - freq_range ** 2) ** 2 + 11723776 * freq_range ** 2)
    n_freq_range = (freq_range / (6.8966888496476 * 10 ** (-5))) * np.sqrt(
        h_freq_range / ((freq_range ** 2 + 79919.29) * (freq_range ** 2 + 1345600)))
    equal_loudness_contour = 20 * np.log10(np.abs(n_freq / n_freq_range))
    return equal_loudness_contour, freq_range


# %%
equal_loudness_contour, freq_range = compute_equal_loudness_contour()

libfmp.b.plot_signal(equal_loudness_contour, T_coef=freq_range, figsize=(6, 3),
                     xlabel='Frequency (Hz)', ylabel='Intensity (dB)',
                     title='Equal Loudness Contour', color='red')
plt.xscale('log')
plt.grid()
plt.show()

# %% [markdown]
# ## Chirp Signal Experiment
#
# A chirp signal with constant intensity sounds louder around 2-4 kHz.
# Adjusting amplitude according to the equal loudness contour produces
# perceptually constant loudness.

# %%
def generate_chirp_exp(dur, freq_start, freq_end, Fs=22050):
    """Generate chirp with exponential frequency increase

    Args:
        dur (float): Duration in seconds
        freq_start (float): Start frequency
        freq_end (float): End frequency
        Fs (scalar): Sampling rate

    Returns:
        x (np.ndarray): Chirp signal
        t (np.ndarray): Time axis
        freq (np.ndarray): Instantaneous frequency
    """
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    freq = np.exp(np.linspace(np.log(freq_start), np.log(freq_end), N))
    phases = np.zeros(N)
    for n in range(1, N):
        phases[n] = phases[n - 1] + 2 * np.pi * freq[n - 1] / Fs
    x = np.sin(phases)
    return x, t, freq


# %%
Fs = 22050
freq_start = 30
freq_end = 10000
dur = 10
x, t, freq = generate_chirp_exp(dur, freq_start, freq_end, Fs=Fs)

print("Chirp with equal intensity (sounds louder around 2-4 kHz):")
display(Audio(x, rate=Fs))

# %%
def generate_chirp_exp_equal_loudness(dur, freq_start, freq_end, Fs=22050):
    """Generate chirp with equal loudness adjustment"""
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    intensity, freq = compute_equal_loudness_contour(freq_min=freq_start, freq_max=freq_end, num_points=N)
    amp = 10 ** (intensity / 20)
    phases = np.zeros(N)
    for n in range(1, N):
        phases[n] = phases[n - 1] + 2 * np.pi * freq[n - 1] / Fs
    x = amp * np.sin(phases)
    return x, t, freq, intensity


x_equal_loudness, t, freq, intensity = generate_chirp_exp_equal_loudness(dur, freq_start, freq_end, Fs=Fs)

print("Chirp with equal loudness adjustment:")
display(Audio(x_equal_loudness, rate=Fs))

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller and Tim Zunner.

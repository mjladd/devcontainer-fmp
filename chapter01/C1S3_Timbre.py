# %% [markdown]
# # Timbre
#
# Following Section 1.3.4 of [Muller, FMP, Springer 2015], we address aspects of
# music related to timbre.

# %% [markdown]
# ## Basic Definition
#
# **Timbre** or **tone color** allows a listener to distinguish between musical
# instruments playing the same pitch at the same loudness. Timbre is subjective
# and can be described with words like bright, dark, warm, or harsh.
#
# Timbre correlates with objective characteristics:
# - Temporal and spectral evolution
# - Tonal and noise-like components
# - Energy distribution across partials

# %% [markdown]
# ## Envelope and ADSR Model
#
# The **envelope** of a waveform is a smooth curve outlining its amplitude extremes.
# In sound synthesis, the **ADSR model** describes four phases:
# - **Attack (A)**: Initial buildup, often with transient noise
# - **Decay (D)**: Sound stabilizes
# - **Sustain (S)**: Steady state with periodic pattern
# - **Release (R)**: Sound fades out

# %%
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa
import os
import sys

sys.path.append('..')
import libfmp.b
import libfmp.c1

%matplotlib inline


def compute_adsr(len_A=10, len_D=10, len_S=60, len_R=10, height_A=1.0, height_S=0.5):
    """Compute idealized ADSR model

    Args:
        len_A (int): Length of Attack phase
        len_D (int): Length of Decay phase
        len_S (int): Length of Sustain phase
        len_R (int): Length of Release phase
        height_A (float): Height of Attack peak
        height_S (float): Height of Sustain level

    Returns:
        curve_ADSR (np.ndarray): ADSR envelope
    """
    curve_A = np.arange(len_A) * height_A / len_A
    curve_D = height_A - np.arange(len_D) * (height_A - height_S) / len_D
    curve_S = np.ones(len_S) * height_S
    curve_R = height_S * (1 - np.arange(1, len_R + 1) / len_R)
    curve_ADSR = np.concatenate((curve_A, curve_D, curve_S, curve_R))
    return curve_ADSR


# %%
# Plot ADSR models with different parameters
curve_ADSR = compute_adsr(len_A=10, len_D=10, len_S=60, len_R=10, height_A=1.0, height_S=0.5)
libfmp.b.plot_signal(curve_ADSR, figsize=(4, 2.5), ylabel='Amplitude', title='ADSR model', color='red')
plt.show()

curve_ADSR = compute_adsr(len_A=20, len_D=2, len_S=60, len_R=1, height_A=2.0, height_S=1.2)
libfmp.b.plot_signal(curve_ADSR, figsize=(4, 2.5), ylabel='Amplitude', title='ADSR model (variant)', color='red')
plt.show()

# %% [markdown]
# ## Envelope Computation
#
# Computing the envelope of a real waveform using a sliding window maximum filter.

# %%
def compute_envelope(x, win_len_sec=0.01, Fs=4000):
    """Compute signal envelopes

    Args:
        x (np.ndarray): Signal to analyze
        win_len_sec (float): Window length in seconds
        Fs (scalar): Sampling rate

    Returns:
        env (np.ndarray): Magnitude envelope
        env_upper (np.ndarray): Upper envelope
        env_lower (np.ndarray): Lower envelope
    """
    win_len_half = round(win_len_sec * Fs * 0.5)
    N = x.shape[0]
    env = np.zeros(N)
    env_upper = np.zeros(N)
    env_lower = np.zeros(N)
    for i in range(N):
        i_start = max(0, i - win_len_half)
        i_end = min(N, i + win_len_half)
        env[i] = np.amax(np.abs(x)[i_start:i_end])
        env_upper[i] = np.amax(x[i_start:i_end])
        env_lower[i] = np.amin(x[i_start:i_end])
    return env, env_upper, env_lower


# %%
# Analyze piano and violin sounds
Fs = 11025
win_len_sec = 0.05

fn_wav = os.path.join('..', 'data', 'C1', 'FMP_C1_F23_Piano.wav')
x_piano, Fs = librosa.load(fn_wav, sr=Fs)
env_piano, env_upper, env_lower = compute_envelope(x_piano, win_len_sec=win_len_sec, Fs=Fs)
t = np.arange(x_piano.size) / Fs

plt.figure(figsize=(8, 3))
plt.plot(t, x_piano, color='gray', label='Waveform')
plt.plot(t, env_piano, linewidth=2, color='red', label='Magnitude envelope')
plt.title('Piano Sound - C4')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()
ipd.display(ipd.Audio(data=x_piano, rate=Fs))

fn_wav = os.path.join('..', 'data', 'C1', 'FMP_C1_F23_Violin.wav')
x_violin, Fs = librosa.load(fn_wav, sr=Fs)
env_violin, env_upper, env_lower = compute_envelope(x_violin, win_len_sec=win_len_sec, Fs=Fs)
t = np.arange(x_violin.size) / Fs

plt.figure(figsize=(8, 3))
plt.plot(t, x_violin, color='gray', label='Waveform')
plt.plot(t, env_violin, linewidth=2, color='red', label='Magnitude envelope')
plt.title('Violin Sound - C4')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()
ipd.display(ipd.Audio(data=x_violin, rate=Fs))

# %% [markdown]
# ## Vibrato and Tremolo
#
# - **Tremolo**: Periodic variations in amplitude (amplitude modulation)
# - **Vibrato**: Periodic variations in frequency (frequency modulation)
#
# These effects influence timbre without necessarily changing perceived pitch/loudness.

# %%
def generate_sinusoid_vibrato(dur=5, Fs=1000, amp=0.5, freq=440, vib_amp=1, vib_rate=5):
    """Generate sinusoid with vibrato (frequency modulation)"""
    num_samples = int(Fs * dur)
    t = np.arange(num_samples) / Fs
    freq_vib = freq + vib_amp * np.sin(t * 2 * np.pi * vib_rate)
    phase_vib = np.zeros(num_samples)
    for i in range(1, num_samples):
        phase_vib[i] = phase_vib[i - 1] + 2 * np.pi * freq_vib[i - 1] / Fs
    x = amp * np.sin(phase_vib)
    return x, t


def generate_sinusoid_tremolo(dur=5, Fs=1000, amp=0.5, freq=440, trem_amp=0.1, trem_rate=5):
    """Generate sinusoid with tremolo (amplitude modulation)"""
    num_samples = int(Fs * dur)
    t = np.arange(num_samples) / Fs
    amps = amp + trem_amp * np.sin(t * 2 * np.pi * trem_rate)
    x = amps * np.sin(2 * np.pi * (freq * t))
    return x, t


# %%
Fs = 4000
dur = 5
freq = 220
amp = 0.5

x, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=Fs, amp=amp, freq=freq)
x_vib, t = generate_sinusoid_vibrato(dur=dur, Fs=Fs, amp=amp, freq=freq, vib_amp=6, vib_rate=5)
x_trem, t = generate_sinusoid_tremolo(dur=dur, Fs=Fs, amp=amp, freq=freq, trem_amp=0.3, trem_rate=5)

print("Pure sinusoid:")
ipd.display(ipd.Audio(data=x, rate=Fs))
print("Sinusoid with vibrato:")
ipd.display(ipd.Audio(data=x_vib, rate=Fs))
print("Sinusoid with tremolo:")
ipd.display(ipd.Audio(data=x_trem, rate=Fs))

# %% [markdown]
# ## Partials and Missing Fundamental
#
# **Partials** are the dominant frequencies of a musical tone, with the lowest
# being the **fundamental frequency**. The relative strengths of partials
# strongly characterize timbre.
#
# **Missing fundamental**: A human can perceive the pitch of a tone even if
# the fundamental frequency is absent, based on the relationships between
# higher harmonics.

# %%
def generate_tone(p=60, weight_harmonic=np.ones([16, 1]), Fs=11025, dur=2):
    """Generate a tone with weighted harmonics

    Args:
        p (float): MIDI pitch
        weight_harmonic (np.ndarray): Weights for harmonics
        Fs (scalar): Sampling rate
        dur (float): Duration in seconds

    Returns:
        x (np.ndarray): Generated signal
        t (np.ndarray): Time axis
    """
    freq = 2 ** ((p - 69) / 12) * 440
    num_samples = int(Fs * dur)
    t = np.arange(num_samples) / Fs
    x = np.zeros(t.shape)
    for h, w in enumerate(weight_harmonic):
        x = x + w * np.sin(2 * np.pi * freq * (h + 1) * t)
    return x, t


# %%
Fs = 11025
p = 60  # C4

print('Pure tone (p = %s):' % p)
x, t = generate_tone(Fs=Fs, p=p, weight_harmonic=[0.2])
ipd.display(ipd.Audio(data=x, rate=Fs))

print('Tone with harmonics (p = %s):' % p)
x, t = generate_tone(Fs=Fs, p=p, weight_harmonic=[0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
ipd.display(ipd.Audio(data=x, rate=Fs))

print('Tone with missing fundamental (p = %s):' % p)
x, t = generate_tone(Fs=Fs, p=p, weight_harmonic=[0, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
ipd.display(ipd.Audio(data=x, rate=Fs))

print('Pure tone at second harmonic (p = %s, one octave higher):' % (p + 12))
x, t = generate_tone(Fs=Fs, p=p, weight_harmonic=[0, 0.2])
ipd.display(ipd.Audio(data=x, rate=Fs))

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller and Tim Zunner.

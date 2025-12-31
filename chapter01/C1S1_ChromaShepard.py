# %% [markdown]
# # Chroma and Shepard Tones
#
# In this notebook, we discuss the notion of chroma and introduce Shepard's helix
# of pitch following Section 1.1.1 of [Muller, FMP, Springer 2015].

# %% [markdown]
# ## Chroma
#
# The term **chroma** closely relates to the twelve different pitch classes.
# Notes that belong to the same pitch class have the same chroma value and are
# perceived as similar. The cyclic nature of chroma values is illustrated by the
# **chromatic circle**. **Shepard's helix of pitch** represents the linear pitch
# space as a helix wrapped around a cylinder.

# %%
import numpy as np
import IPython.display as ipd


def generate_shepard_tone(freq=440, dur=0.5, Fs=44100, amp=1):
    """Generate Shepard tone

    Args:
        freq (float): Frequency of Shepard tone (Default value = 440)
        dur (float): Duration (in seconds) (Default value = 0.5)
        Fs (scalar): Sampling rate (Default value = 44100)
        amp (float): Amplitude of generated signal (Default value = 1)

    Returns:
        x (np.ndarray): Shepard tone
        t (np.ndarray): Time axis (in seconds)
    """
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    num_sin = 1
    x = np.sin(2 * np.pi * freq * t)
    freq_lower = freq / 2
    while freq_lower > 20:
        num_sin += 1
        x = x + np.sin(2 * np.pi * freq_lower * t)
        freq_lower = freq_lower / 2
    freq_upper = freq * 2
    while freq_upper < 20000:
        num_sin += 1
        x = x + np.sin(2 * np.pi * freq_upper * t)
        freq_upper = freq_upper * 2
    x = x / num_sin
    x = amp * x / np.max(x)
    return x, t


def f_pitch(p):
    """Convert MIDI pitch to frequency"""
    F_A4 = 440
    return F_A4 * 2 ** ((p - 69) / 12)


# %% [markdown]
# ## Shepard Tones
#
# Shepard tones are weighted superpositions of sine waves separated by octaves.
# When played moving up the chromatic scale, they create the auditory illusion
# of a tone that continuously moves up (similar to the Penrose stairs illusion).

# %%
# Generate Shepard tones for chromatic scale C3 to C5
Fs = 44100
dur = 0.5

pitch_start = 48
pitch_end = 72
scale = []
for p in range(pitch_start, pitch_end + 1):
    freq = f_pitch(p)
    s, t = generate_shepard_tone(freq=freq, dur=dur, Fs=Fs, amp=0.5)
    scale = np.concatenate((scale, s))

print("Shepard tone chromatic scale (C3 to C5):")
ipd.display(ipd.Audio(scale, rate=Fs))

# %% [markdown]
# ## Shepard-Risset Glissando
#
# The continuous version of Shepard tones, created by Jean-Claude Risset, is
# known as the **Shepard-Risset glissando**. It uses a chirp signal with
# exponentially rising frequency that covers exactly one octave.

# %%
def generate_chirp_exp_octave(freq_start=440, dur=8, Fs=44100, amp=1):
    """Generate one octave of a chirp with exponential frequency increase

    Args:
        freq_start (float): Start frequency of chirp (Default value = 440)
        dur (float): Duration (in seconds) (Default value = 8)
        Fs (scalar): Sampling rate (Default value = 44100)
        amp (float): Amplitude of generated signal (Default value = 1)

    Returns:
        x (np.ndarray): Chirp signal
        t (np.ndarray): Time axis (in seconds)
    """
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    x = np.sin(2 * np.pi * freq_start * np.power(2, t / dur) / np.log(2) * dur)
    x = amp * x / np.max(x)
    return x, t


def generate_shepard_glissando(num_octaves=3, dur_octave=8, Fs=44100):
    """Generate several octaves of a Shepard glissando

    Args:
        num_octaves (int): Number of octaves (Default value = 3)
        dur_octave (int): Duration (in seconds) per octave (Default value = 8)
        Fs (scalar): Sampling rate (Default value = 44100)

    Returns:
        x (np.ndarray): Shepard glissando
        t (np.ndarray): Time axis (in seconds)
    """
    freqs_start = 10 * 2 ** np.arange(0, 11)
    # Generate Shepard glissando by superimposing chirps that differ by octaves
    for freq in freqs_start:
        if freq == 10:
            x, t = generate_chirp_exp_octave(freq_start=freq, dur=dur_octave, Fs=Fs, amp=1)
        else:
            chirp, t = generate_chirp_exp_octave(freq_start=freq, dur=dur_octave, Fs=Fs, amp=1)
            x = x + chirp
    x = x / len(freqs_start)
    # Concatenate several octaves
    x = np.tile(x, num_octaves)
    N = len(x)
    t = np.arange(N) / Fs
    return x, t


# %%
# Generate ascending Shepard-Risset glissando
glissando, t = generate_shepard_glissando(num_octaves=3, dur_octave=8)
print("Shepard-Risset glissando (3 octaves):")
ipd.display(ipd.Audio(glissando, rate=Fs))

# %% [markdown]
# Risset used such a glissando in the second movement "Fall" of his electronic
# music piece "Computer Suite from Little Boy" (1968), making reference to the
# atomic bomb dropped on Hiroshima with seemingly endlessly descending glissandi.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller.

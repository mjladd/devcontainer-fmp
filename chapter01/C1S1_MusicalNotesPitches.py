# %% [markdown]
# # Musical Notes and Pitches
#
# Following Section 1.1.1 of [Muller, FMP, Springer 2015], we introduce the notions
# of musical notes, pitches, and the twelve-tone equal-tempered scale.

# %% [markdown]
# ## Notes and Pitches
#
# In music, the term **note** may refer to both a musical symbol (in score
# representations) and a pitched sound (in audio representations). The notion of
# **pitch** refers to a perceptual property that allows a listener to order a
# sound on a frequency-related scale.

# %% [markdown]
# ## Pitch Class and Octaves
#
# Two notes with fundamental frequencies in a ratio equal to any power of two
# are perceived as very similar. All notes with this relation can be grouped
# under the same **pitch class**. An **octave** is the interval between one
# musical note and another with half or double its fundamental frequency.

# %%
import numpy as np
import IPython.display as ipd


def generate_sinusoid_pitches(pitches=[69], dur=0.5, Fs=4000, amp=1):
    """Generation of sinusoids for a given list of MIDI pitches

    Args:
        pitches (list): List of MIDI pitches (Default value = [69])
        dur (float): Duration (in seconds) of each sinusoid (Default value = 0.5)
        Fs (scalar): Sampling rate (Default value = 4000)
        amp (float): Amplitude of generated signal (Default value = 1)

    Returns:
        x (np.ndarray): Signal
        t (np.ndarray): Time axis (in seconds)
    """
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    x = []
    for p in pitches:
        freq = 2 ** ((p - 69) / 12) * 440
        x = np.append(x, np.sin(2 * np.pi * freq * t))
    x = amp * x / np.max(x)
    return x, t


# %%
# Pitch class C example
dur = 1
Fs = 22050

pitches = [36, 48, 60, 72, 84, 96, 108]
x, t = generate_sinusoid_pitches(pitches=pitches, dur=dur, Fs=Fs, amp=0.5)
print('Pitch class C = {..., C1, C2, C3, C4, C5, C6, C7, ...}', flush=True)
ipd.display(ipd.Audio(data=x, rate=Fs))

# %% [markdown]
# ## Musical Scales
#
# A **musical scale** is a finite set of representative pitches that discretizes
# the space of all possible pitches. Scales are generally considered to span a
# single octave, with higher or lower octaves repeating the pattern.

# %%
# C major and C minor scales
dur = 0.5
Fs = 22050

x_maj, t = generate_sinusoid_pitches(pitches=[60, 62, 64, 65, 67, 69, 71, 72], dur=dur, Fs=Fs, amp=0.5)
x_min, t = generate_sinusoid_pitches(pitches=[60, 62, 63, 65, 67, 68, 70, 72], dur=dur, Fs=Fs, amp=0.5)

print('C major scale', flush=True)
ipd.display(ipd.Audio(data=x_maj, rate=Fs))
print('C minor scale', flush=True)
ipd.display(ipd.Audio(data=x_min, rate=Fs))

# %% [markdown]
# ## Twelve-Tone Equal-Tempered Scale
#
# In the **twelve-tone equal-tempered scale**, an octave is subdivided into twelve
# scale steps. The fundamental frequencies are equally spaced on a logarithmic
# frequency axis. The difference between two subsequent scale steps is called a
# **semitone**.

# %% [markdown]
# ## Enharmonic Equivalence
#
# In the twelve-tone equal-tempered scale, there are twelve pitch classes:
# - Seven denoted by letters: C, D, E, F, G, A, B (white keys)
# - Five denoted with accidentals: C#/Db, D#/Eb, F#/Gb, G#/Ab, A#/Bb (black keys)
#
# **Enharmonic equivalence**: C# and Db represent the same pitch class.

# %% [markdown]
# ## Scientific Pitch Notation
#
# Each note is specified by the pitch class name followed by an octave number.
# The note A4 has a fundamental frequency of 440 Hz and serves as a reference.
#
# Formula for center frequency:
# F_pitch(p) = 2^((p-69)/12) * 440

# %%
# Chromatic scale from C3 to C5
dur = 0.25
Fs = 22050
pitches = range(48, 73)

x_chromatic, t = generate_sinusoid_pitches(pitches=pitches, dur=dur, Fs=Fs, amp=0.5)

print('Sinusoidal sonification of the chromatic scale ranging from C3 (p=48) to C5 (p=72):', flush=True)
ipd.display(ipd.Audio(data=x_chromatic, rate=Fs))

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller.

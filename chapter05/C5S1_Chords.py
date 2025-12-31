# %% [markdown]
# # Chords
#
# Following Section 5.1.2 of [Müller, FMP, Springer 2015], we introduce in this notebook
# some basic facts on musical chords.

# %% [markdown]
# ## Introduction
#
# Intuitively, a **chord** can be loosely defined as a group of several notes that sound
# simultaneously. While most researchers agree that a chord should contain at least three
# notes, others also regard a combination of two notes as a chord. Depending on the number
# of distinct notes contained in a chord, one also speaks of a **dyad** (two notes,
# corresponding to intervals), a **triad** (three notes), a **tetrad** (four notes), and so on.
# In harmony analysis, notes that are one or several octaves apart are often considered to
# belong to the same "sound quality." Therefore, when defining the concept of a chord, it
# may be more precise to speak of distinct pitch classes rather than of distinct notes.

# %% [markdown]
# ## Triads
#
# Despite being an oversimplification, we restrict our considerations in the following to
# a small selection of chords. In Western music, the most important **triads** consist of
# three notes that can be stacked in thirds. When stacked in thirds, the lowest note is
# referred to as the **root note**. Since there are minor thirds (three semitones) and
# major thirds (four semitones), one can distinguish between four types of such triads:
# **major triad**, **minor triad**, **diminished triad**, and **augmented triad**.

# %%
import numpy as np
import IPython.display as ipd

def generate_sinusoid_chord(pitches=[69], duration=1, Fs=4000, amplitude_max=0.5):
    """Generate synthetic sound of chord using sinusoids

    Notebook: C5/C5S1_Chords.ipynb

    Args:
        pitches (list): List of pitches (MIDI note numbers) (Default value = [69])
        duration (float): Duration (seconds) (Default value = 1)
        Fs (scalar): Sampling rate (Default value = 4000)
        amplitude_max (float): Amplitude (Default value = 0.5)

    Returns:
        x (np.ndarray): Synthesized signal
    """
    N = int(duration * Fs)
    t = np.arange(0, N) / Fs
    x = np.zeros(N)
    for p in pitches:
        omega = 2 ** ((p - 69) / 12) * 440
        x = x + np.sin(2 * np.pi * omega * t)
    x = amplitude_max * x / np.max(x)
    return x

duration = 2
Fs = 4000

pitches = [60, 64, 67]
x = generate_sinusoid_chord(pitches=pitches, duration=duration, Fs=Fs)
print('Major chord', flush=True)
ipd.display(ipd.Audio(data=x, rate=Fs))

pitches = [60, 63, 67]
x = generate_sinusoid_chord(pitches=pitches, duration=duration, Fs=Fs)
print('Minor chord', flush=True)
ipd.display(ipd.Audio(data=x, rate=Fs))

pitches = [60, 63, 66]
x = generate_sinusoid_chord(pitches=pitches, duration=duration, Fs=Fs)
print('Diminished chord', flush=True)
ipd.display(ipd.Audio(data=x, rate=Fs))

pitches = [60, 64, 68]
x = generate_sinusoid_chord(pitches=pitches, duration=duration, Fs=Fs)
print('Augmented chord', flush=True)
ipd.display(ipd.Audio(data=x, rate=Fs))

# %% [markdown]
# ## Major and Minor Chords
#
# Since there are twelve different root notes (up to enharmonic equivalence and octave
# shifts), one can basically form twelve major and twelve minor triads.

# %%
duration = 1
Fs = 4000

x_major = []
pitches = np.array([60, 64, 67])
for i in range(12):
    x = generate_sinusoid_chord(pitches=pitches+i, duration=duration, Fs=Fs)
    x_major = np.append(x_major, x)

x_minor = []
pitches = np.array([60, 63, 67])
for i in range(12):
    x = generate_sinusoid_chord(pitches=pitches+i, duration=duration, Fs=Fs)
    x_minor = np.append(x_minor, x)

print('Major chords', flush=True)
ipd.display(ipd.Audio(data=x_major, rate=Fs))

print('Minor chords', flush=True)
ipd.display(ipd.Audio(data=x_minor, rate=Fs))

# %% [markdown]
# On the note level, there are generally many alternatives for realizing a given chord.
# When a chord's lowest note (the bass note) is its root, the chord is said to be in
# **root position** or in **normal form**. When the root is not the lowest note played
# in a chord, it is said to be **inverted**.

# %%
Fs = 4000
duration = 2
x = []
x = generate_sinusoid_chord(pitches=[60, 64, 67], duration=duration, Fs=Fs)
x = np.append(x,generate_sinusoid_chord(pitches=[64, 67, 72], duration=duration, Fs=Fs))
x = np.append(x,generate_sinusoid_chord(pitches=[67, 72, 76], duration=duration, Fs=Fs))
x = np.append(x,generate_sinusoid_chord(pitches=[60, 64, 67, 72, 76], duration=duration, Fs=Fs))
duration = duration / 8
broken_chord = [60, 64, 67, 72, 76, 67, 72, 76]
for p in broken_chord:
    x = np.append(x,generate_sinusoid_chord(pitches=[p], duration=duration, Fs=Fs))

ipd.display(ipd.Audio(data=x, rate=Fs))

# %% [markdown]
# A major chord is usually denoted with the same symbol as used for the pitch class of
# its root note. For the minor chords, one often uses the same notation as for major
# chords except for adding a letter **m** that refers to "minor."

# %% [markdown]
# ## Mathematical Model
#
# Adopting a rather simplistic view, where a major or minor chord is determined by the
# **pitch classes** or **chroma values** of its constituent notes, one can regard each
# of the triads as a three-element subset of the set {C, C#, D, ..., B} that consists
# of the twelve chroma attributes. Based on this mathematical model, the twelve major
# chords can be obtained by cyclically shifting the major triad C in twelve different
# ways. Similarly, one obtains the twelve minor chords from Cm.

# %% [markdown]
# ## Seventh Chords and Beyond
#
# The triads also appear as subsets of more complex chords consisting of four, five or
# even more notes. Stacking one further third interval onto a triad leads to a
# **seventh chord**. There are many possible types of seventh chords including:
# - **major seventh chord** (major triad + major third)
# - **dominant seventh chord** (major triad + minor third)
# - **minor seventh chord** (minor triad + minor third)
# - **half-diminished seventh chord** (diminished triad + major third)
# - **diminished seventh chord** (diminished triad + minor third)

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Christof Weiß.

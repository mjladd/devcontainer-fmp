# %% [markdown]
# # Harmonic Series
#
# In this notebook we look at the harmonic series following Section 1.3.2 of
# [Muller, FMP, Springer 2015].

# %% [markdown]
# ## Definition
#
# Let w denote the center frequency of a musical note (e.g., C2 with w=65.4 Hz).
# The **harmonic series** is an arithmetic series: w, 2w, 3w, 4w, ...
#
# Since pitch perception is logarithmic in frequency, higher harmonics appear
# "closer together" than lower ones. This differs from the **octave series**
# (geometric progression: w, 2w, 4w, 8w, ...) where each interval sounds "the same."
#
# For C2 (w=65.4 Hz):
# - 2nd harmonic (2w) = C3 (one octave higher)
# - 3rd harmonic (3w) = G3 (perfect fifth above C3)
# - 4th harmonic (4w) = C4 (two octaves higher)

# %%
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import pandas as pd
from collections import OrderedDict
import sys

sys.path.append('..')
import libfmp.c1

# %%
# Compute harmonic frequencies and compare with equal-tempered notes
p = 36  # C2
freq = libfmp.c1.f_pitch(p)
freq_harmonic = (np.asarray(range(16)) + 1) * freq

# Notes closest to each harmonic
notes = np.asarray([36, 48, 55, 60, 64, 67, 70, 72, 74, 76, 78, 79, 80, 82, 83, 84])
freq_center = libfmp.c1.f_pitch(notes)
freq_deviation_cents = libfmp.c1.difference_cents(freq_harmonic, freq_center)

# Generate sinusoids
dur = 4  # seconds
Fs = 4000  # sampling rate

sinusoid_freq_center = []
for f in freq_center:
    x, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=Fs, freq=f)
    sinusoid_freq_center.append(x)

sinusoid_freq_harmonic = []
for f in freq_harmonic:
    x, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=Fs, freq=f)
    sinusoid_freq_harmonic.append(x)

# %%
# Display comparison table
note_names = ['C2', 'C3', 'G3', 'C4', 'E4', 'G4', 'Bb4', 'C5', 'D5', 'E5',
              'F#5', 'G5', 'Ab5', 'Bb5', 'B5', 'C6']

print("Harmonic Series starting from C2 (MIDI 36, 65.4 Hz)")
print("=" * 70)
print(f"{'Harm.':<6} {'Note':<6} {'Note Freq':<12} {'Harm. Freq':<12} {'Deviation':<12}")
print("-" * 70)
for i in range(16):
    print(f"{i+1:<6} {note_names[i]:<6} {freq_center[i]:<12.2f} {freq_harmonic[i]:<12.2f} {freq_deviation_cents[i]:+.2f} cents")

# %%
# Play individual harmonics
print("\nHarmonic sinusoids:")
for i in [0, 2, 6, 10]:  # 1st, 3rd, 7th, 11th harmonics
    print(f"Harmonic {i+1} ({note_names[i]}, {freq_harmonic[i]:.1f} Hz):")
    ipd.display(ipd.Audio(data=sinusoid_freq_harmonic[i], rate=Fs))

# %% [markdown]
# ## Superposition of Harmonics
#
# When sinusoids with frequencies from the harmonic series are superimposed,
# the result is perceived as a single, homogeneous sound. In contrast,
# superimposing sinusoids at equal-tempered note frequencies produces a
# more heterogeneous sound.

# %%
# Superimpose all 16 harmonics
num_sinusoid = 16
x_all_harmonic = sinusoid_freq_harmonic[0].copy()
x_all_center = sinusoid_freq_center[0].copy()
for i in range(num_sinusoid - 1):
    x_all_harmonic = x_all_harmonic + sinusoid_freq_harmonic[i + 1]
    x_all_center = x_all_center + sinusoid_freq_center[i + 1]

x_all_harmonic = x_all_harmonic / num_sinusoid
x_all_center = x_all_center / num_sinusoid

print('Superposition of sinusoids with harmonic frequencies (homogeneous):')
ipd.display(ipd.Audio(data=x_all_harmonic, rate=Fs))
print('Superposition of sinusoids with equal-tempered note frequencies (heterogeneous):')
ipd.display(ipd.Audio(data=x_all_center, rate=Fs))

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller, Frank Zalkow,
# and Shrishti Shetu.

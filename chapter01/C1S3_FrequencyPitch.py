# %% [markdown]
# # Frequency and Pitch
#
# Following Section 1.3.2 of [Muller, FMP, Springer 2015], we cover the relation
# between frequency and pitch.

# %% [markdown]
# ## Sinusoids
#
# A **sinusoid** is the simplest periodic waveform, completely specified by:
# - **Frequency**: measured in Hertz (Hz), reciprocal of period
# - **Amplitude**: peak deviation from mean
# - **Phase**: position in cycle at time zero

# %% [markdown]
# ## Audible Frequency Range
#
# The audible frequency range for humans is approximately 20 Hz to 20,000 Hz.
# Other species have different ranges (dogs: ~45 kHz, cats: ~64 kHz, bats: >100 kHz).

# %%
import IPython.display as ipd
import numpy as np
import sys

sys.path.append('..')
import libfmp.c1

# %%
# Generate chirp signal rising one octave per second (80 Hz to 20480 Hz)
Fs = 44100
dur = 1
freq_start = 80 * 2 ** np.arange(8)
for f in freq_start:
    if f == freq_start[0]:
        chirp, t = libfmp.c1.generate_chirp_exp_octave(freq_start=f, dur=dur, Fs=Fs, amp=1)
    else:
        chirp_oct, t = libfmp.c1.generate_chirp_exp_octave(freq_start=f, dur=dur, Fs=Fs, amp=1)
        chirp = np.concatenate((chirp, chirp_oct))

print("Chirp: 80 Hz to 20480 Hz (one octave per second)")
ipd.display(ipd.Audio(chirp, rate=Fs))

# %%
# Generate descending chirp signal (640 Hz to 20 Hz)
Fs = 8000
dur = 2
freq_start = 20 * 2 ** np.arange(5)
for f in freq_start:
    if f == freq_start[0]:
        chirp, t = libfmp.c1.generate_chirp_exp_octave(freq_start=f, dur=dur, Fs=Fs, amp=1)
    else:
        chirp_oct, t = libfmp.c1.generate_chirp_exp_octave(freq_start=f, dur=dur, Fs=Fs, amp=1)
        chirp = np.concatenate((chirp, chirp_oct))

chirp = chirp[::-1]
print("Descending chirp: 640 Hz to 20 Hz")
ipd.display(ipd.Audio(chirp, rate=Fs))

# %% [markdown]
# ## Pitches and Center Frequencies
#
# The notion of frequency is closely related to **pitch**. In the case of pure
# tones, a sinusoid of 440 Hz corresponds to pitch A4 (concert pitch).
#
# Two frequencies are perceived as similar if they differ by a power of two
# (defining an **octave**). Human perception of pitch is logarithmic in nature.
#
# Using MIDI note numbers, the **center frequency** is defined as:
#
# F_pitch(p) = 2^((p-69)/12) * 440
#
# where p=69 corresponds to A4 (440 Hz).

# %%
def f_pitch(p):
    """Compute center frequency for MIDI note numbers

    Args:
        p (float or np.ndarray): MIDI note numbers

    Returns:
        freq_center (float or np.ndarray): Center frequency
    """
    freq_center = 2 ** ((p - 69) / 12) * 440
    return freq_center


chroma = ['A ', 'A#', 'B ', 'C ', 'C#', 'D ', 'D#', 'E ', 'F ', 'F#', 'G ', 'G#']

print("MIDI note numbers and frequencies for piano keys (A0 to C8):")
for p in range(21, 109):
    print('p = %3d (%2s%1d), freq = %7.2f Hz' % (p, chroma[(p - 69) % 12], (p // 12 - 1), f_pitch(p)))

# %% [markdown]
# ## Cents
#
# The **cent** is a logarithmic unit for musical intervals. An octave = 1200 cents,
# a semitone = 100 cents.
#
# The difference in cents between frequencies w1 and w2:
# log2(w1/w2) * 1200
#
# The **just noticeable difference** varies by person - typically 10-25 cents.

# %%
def difference_cents(freq_1, freq_2):
    """Difference between two frequencies in cents"""
    delta = np.log2(freq_1 / freq_2) * 1200
    return delta


def generate_sinusoid(dur=5, Fs=1000, amp=1, freq=1, phase=0):
    """Generate a sinusoid signal

    Args:
        dur (float): Duration in seconds
        Fs (scalar): Sampling rate
        amp (float): Amplitude
        freq (float): Frequency
        phase (float): Phase

    Returns:
        x (np.ndarray): Signal
        t (np.ndarray): Time axis
    """
    num_samples = int(Fs * dur)
    t = np.arange(num_samples) / Fs
    x = amp * np.sin(2 * np.pi * (freq * t - phase))
    return x, t


# %%
# Demonstrate just noticeable difference
dur = 5
Fs = 4000
pitch = 69
ref = f_pitch(pitch)
freq_list = ref + np.array([0, 2, 5, 10, ref])

for freq in freq_list:
    x, t = generate_sinusoid(dur=dur, Fs=Fs, freq=freq)
    print('freq = %0.1f Hz (A4 + %0.2f cents)' % (freq, difference_cents(freq, ref)))
    ipd.display(ipd.Audio(data=x, rate=Fs))

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller and Stefan Balke.

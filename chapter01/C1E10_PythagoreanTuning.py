# %% [markdown]
# # Pythagorean Tuning
#
# Following Exercise 1.10 and Section 5.1.1.2 of [Muller, FMP, Springer 2015],
# we discuss the tuning system introduced by Pythagoras and the Pythagorean comma.

# %% [markdown]
# ## Pythagorean Comma
#
# **Pythagorean tuning** (6th century BC) is based on frequency ratios using only
# the ratio 3:2 (the **perfect fifth**).
#
# Starting with a root note frequency w, we:
# 1. Multiply by 3/2 (go up a fifth)
# 2. Divide by 2 if necessary to stay within one octave
# 3. Repeat to produce 13 frequency values
#
# The 13th ratio is the **Pythagorean comma** - indicating the inconsistency
# when trying to define a twelve-tone scale using only perfect fifths.

# %%
import numpy as np
import IPython.display as ipd
import sys

sys.path.append('..')
import libfmp.c1


def compute_ratio(num3, num2):
    """Compute frequency ratio from powers of 3 and 2"""
    return (3 ** num3) / (2 ** num2)


# %%
# Construct the 13 frequency ratios
num3 = 0
num2 = 0
note = 0
diff = 0
s = np.zeros((13, 6))
s[0] = [0, note, num3, num2, compute_ratio(num3, num2), diff]

for m in range(1, 13):
    note = (note + 7) % 12
    if note == 0:
        note = 12
    num3 = num3 + 1
    num2 = num2 + 1
    ratio = compute_ratio(num3, num2)
    if ratio > 2:
        num2 = num2 + 1
        ratio = compute_ratio(num3, num2)
    diff = (np.log2(ratio) - 1 / 12) * 1200
    diff = np.remainder(diff, 100)
    s[m] = [m, note, num3, num2, compute_ratio(num3, num2), diff]

print("Pythagorean ratios constructed by stacking fifths:")
print("=" * 75)
for m in range(13):
    print('m = %2i, note = %2i, ratio = %6i:%6i = %7.4f, diff from ET = %+6.2f cents'
          % (s[m, 0], s[m, 1], 3 ** int(s[m, 2]), 2 ** int(s[m, 3]), s[m, 4], s[m, 5]))

print('\nPythagorean comma: %7.4f (%+6.2f cents)' % (s[12, 4], s[12, 5]))

# %%
# Compare sinusoids at A4 and A4 + Pythagorean comma
dur = 4  # seconds
Fs = 4000  # sampling rate
freq = 440
x, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=Fs, freq=freq)
freq_pyt_comma = freq * s[12, 4]
x_pyt_comma, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=Fs, freq=freq_pyt_comma)

print()
print('Sinusoid of 440 Hz (A4):')
ipd.display(ipd.Audio(data=x, rate=Fs))
print('Sinusoid with %.4f * 440 = %.4f Hz:' % (s[12, 4], freq_pyt_comma))
ipd.display(ipd.Audio(data=x_pyt_comma, rate=Fs))

# %% [markdown]
# ## Pythagorean Scale
#
# When allowing both adding and subtracting fifths and octaves, we obtain the
# **Pythagorean scale**. All intervals can be expressed by frequency ratios
# involving only powers of two and three.

# %%
import pandas as pd
from collections import OrderedDict

# Pythagorean intervals for chromatic scale
pyt_frac = ['1:1', '2^8:3^5', '3^2:2^3', '2^5:3^3', '3^4:2^6', '2^2:3', '3^6:2^9',
            '3:2', '2^7:3^4', '3^3:2^4', '2^4:3^2', '3^5:2^7', '2:1']
pyt_ratio = np.asarray([1, 256 / 243, 9 / 8, 32 / 27, 81 / 64, 4 / 3, 729 / 512,
                        3 / 2, 128 / 81, 27 / 16, 16 / 9, 243 / 128, 2])

p = 60  # C4
freq = libfmp.c1.f_pitch(p)
freq_pyt = pyt_ratio * freq
notes = np.asarray(range(p, p + 13))
freq_center = libfmp.c1.f_pitch(notes)
freq_deviation_cents = libfmp.c1.difference_cents(freq_pyt, freq_center)

# %%
# Generate sinusoids for comparison
dur = 4  # seconds
Fs = 4000  # sampling rate

sinusoid_freq_center = []
for f in freq_center:
    x, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=Fs, freq=f)
    sinusoid_freq_center.append(x)

sinusoid_freq_pyt = []
for f in freq_pyt:
    x, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=Fs, freq=f)
    sinusoid_freq_pyt.append(x)

# %%
# Display comparison
note_names = ['C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5']

print("Pythagorean Scale vs Equal Temperament (starting from C4)")
print("=" * 80)
print(f"{'Note':<6} {'ET Freq':<12} {'Pyt Ratio':<12} {'Pyt Freq':<12} {'Difference':<12}")
print("-" * 80)
for i in range(13):
    print(f"{note_names[i]:<6} {freq_center[i]:<12.2f} {pyt_frac[i]:<12} {freq_pyt[i]:<12.2f} {freq_deviation_cents[i]:+.2f} cents")

# %%
# Play some comparison tones
print("\nEqual Temperament vs Pythagorean - selected notes:")
for i in [0, 4, 7, 12]:  # C, E, G, C (next octave)
    print(f"\n{note_names[i]} - Equal Temperament ({freq_center[i]:.2f} Hz):")
    ipd.display(ipd.Audio(data=sinusoid_freq_center[i], rate=Fs))
    print(f"{note_names[i]} - Pythagorean ({freq_pyt[i]:.2f} Hz, {freq_deviation_cents[i]:+.2f} cents):")
    ipd.display(ipd.Audio(data=sinusoid_freq_pyt[i], rate=Fs))

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller, Frank Zalkow,
# and Shrishti Shetu.

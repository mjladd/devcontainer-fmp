# %% [markdown]
# # Intervals
#
# Following Section 5.1.1 of [Müller, FMP, Springer 2015], we introduce in this notebook
# some basic facts on musical intervals.

# %% [markdown]
# ## Introduction
#
# In music, an **interval** may be loosely defined as the difference between two
# **pitches**. This definition is problematic in the sense that the underlying notion
# of pitch is already a rather vague one. Pitch is a perceptual property that allows
# the ordering of sounds on a frequency-related logarithmic scale.

# %% [markdown]
# ## Semitone Differences
#
# Assuming a **twelve-tone equal-tempered scale**, an octave is subdivided into twelve
# scale steps that are equally spaced on a logarithmic frequency axis. The smallest
# possible interval in this scale is called a **semitone**, which is the difference
# between two subsequent scale steps.

# %% [markdown]
# ## Frequency Ratios
#
# The concept of intervals can be approached from a physical point of view by considering
# frequency relations that naturally occur between the harmonic partials of a pitched sound.

# %%
from collections import OrderedDict
import numpy as np
import IPython.display as ipd
import pandas as pd

def f_pitch(p):
    frequency = 2 ** ((p - 69) / 12) * 440
    return frequency

diff_semitones = ['0','1','2','3','4','5','6','7','8','9','10','11','12',]

JI_frac = ['$1:1$', '$15:16$', '$8:9$', '$5:6$', '$4:5$', '$3:4$', '$32:45$',
           '$2:3$', '$5:8$', '$3:5$', '$5:9$', '$8:15$', '$1:2$']
JI_ratio = np.asarray([1, 15/16, 8/9, 5/6, 4/5, 3/4, 32/45, 2/3, 5/8, 3/5, 5/9, 8/15, 1/2])

pyt_frac = ['$1:1$', '$3^5:2^8$', '$2^3:3^2$', '$3^3:2^5$', '$2^6:3^4$', '$3:2^2$', '$2^9:3^6$',
            '$2:3$', '$3^4:2^7$', '$2^4:3^3$', '$3^2:2^4$', '$2^7:3^5$', '$1:2$']
pyt_ratio = np.asarray([1, 243/256, 8/9, 27/32, 64/81, 3/4, 512/729, 2/3, 81/128, 16/27, 9/16, 128/243, 1/2])

p = 60
omega = f_pitch(p)
freq_JI = omega / JI_ratio
freq_pyt = omega / pyt_ratio
notes = np.asarray(range(p, p + 13))
freq_center = f_pitch(notes)
freq_deviation_cents_JI =  np.log2(freq_JI / freq_center) * 1200
freq_deviation_cents_pyt =  np.log2(freq_pyt / freq_center) * 1200

duration = 1
Fs = 4000
N = int(duration * Fs)
t = np.arange(0, N) / Fs

def generate_sinusoid(omega, t):
    return np.sin(2 * np.pi * omega * t)

def generate_sinusoid_list(freq_list, x_ref=[]):
    sinusoid_list = []
    for f in freq_list:
        s = generate_sinusoid(f, t)
        x = np.concatenate((x_ref, s))
        sinusoid_list.append(x)
    return sinusoid_list

x_ref = generate_sinusoid(omega, t)
sinusoid_center = generate_sinusoid_list(freq_center, x_ref)
sinusoid_JI = generate_sinusoid_list(freq_JI, x_ref)
sinusoid_pyt = generate_sinusoid_list(freq_pyt, x_ref)

# Generation of html table
def generate_audio_tag_html_list(sinusoid_list, Fs):
    audio_tag_html_list = []
    for i in range(len(sinusoid_list)):
        audio_tag = ipd.Audio( sinusoid_list[i], rate=Fs)
        audio_tag_html = audio_tag._repr_html_().replace('\n', '').strip()
        audio_tag_html = audio_tag_html.replace('<audio ',
                                                '<audio style="width: 110px; height: 30px;"')
        audio_tag_html_list.append(audio_tag_html)
    return audio_tag_html_list

audio_tag_html_center = generate_audio_tag_html_list(sinusoid_center, Fs=Fs)
audio_tag_html_JI = generate_audio_tag_html_list(sinusoid_JI, Fs=Fs)
audio_tag_html_pyt = generate_audio_tag_html_list(sinusoid_pyt, Fs=Fs)


pd.options.display.float_format = '{:,.1f}'.format
pd.set_option('display.max_colwidth', None)
df = pd.DataFrame(OrderedDict([
    ('$\Delta$', diff_semitones),
    ('Interval name', ['(Perfect) unison','Minor second','Major second','Minor Third',
                       'Major Third','(Perfect) fourth', 'Tritone','(Perfect) fifth',
                       'Minor sixth','Major sixth','Minor seventh','Major seventh','(Perfect) octave']),
    ('&emsp;Interval', ['C4&ndash;C4','C4&ndash;C$^\sharp$4','C4&ndash;D4',
                        'C4&ndash;D$^\sharp$4','C4&ndash;E4','C4&ndash;F4',
                        'C4&ndash;F$^\sharp$4','C4&ndash;G4','C4&ndash;G$^\sharp$4',
                        'C4&ndash;A4','C4&ndash;A$^\sharp$4', 'C4&ndash;B4','C4&ndash;C4']),
    ('ET Sinusoid', audio_tag_html_center),
    ('&emsp;&emsp; JI Ratio', JI_frac),
    ('JI Sinusoid', audio_tag_html_JI),
    ('JI Dev', freq_deviation_cents_JI),
    ('&emsp;&emsp; Pyt Ratio', pyt_frac),
    ('Pyt Sinusoid', audio_tag_html_pyt),
    ('Pyt Dev', freq_deviation_cents_pyt)]))

df.index = np.arange(1, len(df) + 1)
ipd.HTML(df.to_html(escape=False, index=False))

# %% [markdown]
# ## Consonance and Dissonance
#
# Using the just intonation based on harmonic partials, we have seen that certain intervals
# can be described by ratios of small integers such as 1:1 (unison), 1:2 (octave),
# 2:3 (fifth), or 3:4 (fourth). Such intervals are usually perceived as coherent and
# pleasant. The term **consonance** refers to a combination of notes that sound pleasant
# to most people when being played simultaneously. In contrast, the term **dissonance**
# is used to refer to a combination of notes that sound harsh or unpleasant.

# %%
duration = 3
Fs = 4000
N = int(duration * Fs)
t = np.arange(0, N) / Fs

def generate_sinusoid_interval_list(freq_list, x_ref=[]):
    sinusoid_list = []
    for f in freq_list:
        s = generate_sinusoid(f, t)
        x = x_ref + s
        sinusoid_list.append(x)
    return sinusoid_list

x_ref = generate_sinusoid(omega, t)
sinusoid_center = generate_sinusoid_interval_list(freq_center, x_ref)
sinusoid_JI = generate_sinusoid_interval_list(freq_JI, x_ref)
sinusoid_pyt = generate_sinusoid_interval_list(freq_pyt, x_ref)
sinusoid_sum = list(range(len(freq_center)))
for i in range(len(freq_center)):
    sinusoid_sum[i] = (sinusoid_center[i] + sinusoid_JI[i] + sinusoid_pyt[i]) / 3

# Generation of html table
def generate_audio_tag_html_list(sinusoid_freq_list, Fs):
    audio_tag_html_list = []
    for i in range(len(sinusoid_freq_list)):
        audio_tag = ipd.Audio( sinusoid_freq_list[i], rate=Fs)
        audio_tag_html = audio_tag._repr_html_().replace('\n', '').strip()
        audio_tag_html = audio_tag_html.replace('<audio ',
                                                '<audio style="width: 110px; height: 30px;"')
        audio_tag_html_list.append(audio_tag_html)
    return audio_tag_html_list

audio_tag_html_center = generate_audio_tag_html_list(sinusoid_center, Fs=Fs)
audio_tag_html_JI = generate_audio_tag_html_list(sinusoid_JI, Fs=Fs)
audio_tag_html_pyt = generate_audio_tag_html_list(sinusoid_pyt, Fs=Fs)

pd.options.display.float_format = '{:,.1f}'.format
pd.set_option('display.max_colwidth', None)
df = pd.DataFrame(OrderedDict([
    ('Delta', diff_semitones),
    ('Interval name', ['(Perfect) unison','Minor second','Major second','Minor Third',
                       'Major Third','(Perfect) fourth', 'Tritone','(Perfect) fifth',
                       'Minor sixth','Major sixth','Minor seventh','Major seventh','(Perfect) octave']),
    ('&emsp;Interval', ['C4&ndash;C4','C4&ndash;C$^\sharp$4','C4&ndash;D4',
                        'C4&ndash;D$^\sharp$4','C4&ndash;E4','C4&ndash;F4',
                        'C4&ndash;F$^\sharp$4','C4&ndash;G4','C4&ndash;G$^\sharp$4',
                        'C4&ndash;A4','C4&ndash;A$^\sharp$4', 'C4&ndash;B4','C4&ndash;C4']),
    ('ET Sinusoid', audio_tag_html_center),
    ('&emsp;&emsp; JI Ratio ', JI_frac),
    ('JI Sinusoid', audio_tag_html_JI),
    ('JI Dev.', freq_deviation_cents_JI),
    ('&emsp;&emsp; Pyt Ratio ', pyt_frac),
    ('Pyt Sinusoid', audio_tag_html_pyt),
    ('Pyt Dev.', freq_deviation_cents_pyt)]))

df.index = np.arange(1, len(df) + 1)
ipd.HTML(df.to_html(escape=False, justify='center', index=False))

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

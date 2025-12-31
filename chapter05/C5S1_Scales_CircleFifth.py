# %% [markdown]
# # Musical Scales and Circle of Fifths
#
# Following Section 5.1.2 of [Müller, FMP, Springer 2015], we introduce in this notebook
# some basic facts on musical scales and the circle of fifths.

# %% [markdown]
# ## Introduction
#
# Besides intervals and chords, we now consider another important musical construct that
# is referred to as a **musical scale**. Again, adopting a rather simplistic view, a scale
# can be regarded as a set of notes, where the elements are typically ordered by ascending
# pitch. While a **chord** may be thought of as a **vertical structure**, a **scale** is
# usually associated to **horizontal structures**.

# %% [markdown]
# ## Chromatic Scale
#
# As first example, we consider the twelve-tone equal-tempered scale, where an octave is
# subdivided into twelve scale steps. This scale is also referred to as **chromatic scale**.

# %%
import numpy as np
import IPython.display as ipd

def generate_sinusoid_scale(pitches=[69], duration=0.5, Fs=4000, amplitude_max=0.5):
    """Generate synthetic sound of scale using sinusoids

    Notebook: C5/C5S1_Scales_CircleFifth.ipynb

    Args:
        pitches (list): List of pitchs (MIDI note numbers) (Default value = [69])
        duration (float): Duration (seconds) (Default value = 0.5)
        Fs (scalar): Sampling rate (Default value = 4000)
        amplitude_max (float): Amplitude (Default value = 0.5)

    Returns:
        x (np.ndarray): Synthesized signal
    """
    N = int(duration * Fs)
    t = np.arange(0, N) / Fs
    x = []
    for p in pitches:
        omega = 2 ** ((p - 69) / 12) * 440
        x = np.append(x, np.sin(2 * np.pi * omega * t))
    x = amplitude_max * x / np.max(x)
    return x

duration = 0.25
Fs = 4000
pitches = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]
x = generate_sinusoid_scale(pitches=pitches, duration=duration, Fs=Fs)
print('Chromatic scale', flush=True)
ipd.display(ipd.Audio(data=x, rate=Fs))

# %% [markdown]
# ## Major and Minor Scale
#
# The first scale type is known as a **major scale**, which is made up of seven notes
# and a repeated octave. The second scale type we consider is known as the (natural)
# **minor scale**. Both major and minor scales can be subsumed under the general term
# **diatonic scale**.

# %%
duration = 0.5

x_maj = generate_sinusoid_scale(pitches=[60, 62, 64, 65, 67, 69, 71, 72], duration=duration, Fs=Fs)
x_min = generate_sinusoid_scale(pitches=[60, 62, 63, 65, 67, 68, 70, 72], duration=duration, Fs=Fs)

print('C-major scale', flush=True)
ipd.display(ipd.Audio(data=x_maj, rate=Fs))
print('C-minor scale (natural)', flush=True)
ipd.display(ipd.Audio(data=x_min, rate=Fs))

# %% [markdown]
# ## Further Scales
#
# There are many more scales used in Western music and beyond. For example, besides the
# minor scale introduced above (also referred to as the **natural minor** scale), there
# are other types of minor scales called the **harmonic minor** and **melodic minor** scale.

# %%
duration = 0.5

x_mh = generate_sinusoid_scale(pitches=[60, 62, 63, 65, 67, 68, 71, 72], duration=duration, Fs=Fs)
x_p = generate_sinusoid_scale(pitches=[60, 62, 64, 67, 69, 72], duration=duration, Fs=Fs)
x_w = generate_sinusoid_scale(pitches=[60, 62, 64, 66, 68, 70, 72], duration=duration, Fs=Fs)
x_o = generate_sinusoid_scale(pitches=[60, 61, 63, 64, 66, 67, 69, 70, 72], duration=duration, Fs=Fs)

print('Harmonic minor scale (7 pitches + octave)', flush=True)
ipd.display(ipd.Audio(data=x_mh, rate=Fs))
print('Pentatonic scale (5 pitches + octave)', flush=True)
ipd.display(ipd.Audio(data=x_p, rate=Fs))
print('Whole tone scale (6 pitches + octave)', flush=True)
ipd.display(ipd.Audio(data=x_w, rate=Fs))
print('Octatonic scale (8 pitches + octave)', flush=True)
ipd.display(ipd.Audio(data=x_o, rate=Fs))

# %% [markdown]
# ## Circle of Fifths
#
# One characteristic property of diatonic scales is that they can be obtained from a chain
# of six successive perfect fifth intervals. The famous **circle of fifths** is a visual
# representation of the relationships among the twelve tones of the chromatic scale and
# the associated major and minor scales.

# %% [markdown]
# ## Musical Keys
#
# The circle of fifths represents the relations between musical **keys**—a concept that
# is closely connected to major and minor scales.

# %%
import pandas as pd
from collections import OrderedDict

duration = 0.25

scale_major = np.array([60, 62, 64, 65, 67, 69, 71, 72])
scale_minor = np.array([57, 59, 60, 62, 64, 65, 67, 69])
scale_major_name = ['C','G','D','A','E','B','F$^\sharp$',
                    'D$^\\flat$','A$^\\flat$','E$^\\flat$','B$^\\flat$','F','C',]
scale_minor_name = ['Am','Em','Bm','F$^\sharp$m','C$^\sharp$m','G$^\sharp$m','D$^\sharp$m',
                    'B$^\\flat$m','Fm','Cm','Gm','Dm','Am',]

scale_major_list = []
for i in range(13):
    x = generate_sinusoid_scale(pitches=scale_major, duration=duration, Fs=Fs)
    scale_major_list.append(x)
    scale_major += 7
    if scale_major[-1] > 80:
        scale_major -= 12

scale_minor_list = []
for i in range(13):
    x = generate_sinusoid_scale(pitches=scale_minor, duration=duration, Fs=Fs)
    scale_minor_list.append(x)
    scale_minor += 7
    if scale_minor[-1] > 80:
        scale_minor -= 12


audio_tag_html_list_major = []
for i in range(13):
    audio_tag = ipd.Audio(scale_major_list[i], rate=Fs)
    audio_tag_html = audio_tag._repr_html_().replace('\n', '').strip()
    audio_tag_html = audio_tag_html.replace('<audio ',
                                            '<audio style="width: 200px; height: 30px;"')
    audio_tag_html_list_major.append(audio_tag_html)

audio_tag_html_list_minor = []
for i in range(13):
    audio_tag = ipd.Audio(scale_minor_list[i], rate=Fs)
    audio_tag_html = audio_tag._repr_html_().replace('\n', '').strip()
    audio_tag_html = audio_tag_html.replace('<audio ',
                                            '<audio style="width: 200px; height: 30px;"')
    audio_tag_html_list_minor.append(audio_tag_html)

pd.options.display.float_format = '{:,.1f}'.format
pd.set_option('display.max_colwidth', None)
df = pd.DataFrame(OrderedDict([
    ('Major', scale_major_name),
    (' ', audio_tag_html_list_major),
    ('Minor', scale_minor_name),
    ('  ', audio_tag_html_list_minor)]))

#df.index = np.arange(0, len(df))
#df = df.T
ipd.HTML(df.to_html(escape=False, justify='center', index=True, header=True))

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Christof Weiß.

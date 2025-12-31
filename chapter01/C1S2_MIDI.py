# %% [markdown]
# # Symbolic Format: MIDI
#
# Following Section 1.2.2 of [Muller, FMP, Springer 2015], we look at the MIDI
# standard, which is often used to encode symbolic music.

# %% [markdown]
# ## MIDI Standard
#
# **MIDI** (Musical Instrument Digital Interface) was developed as an industry
# standard to get digital electronic instruments from different manufacturers
# to work together. MIDI allows a musician to remotely and automatically control
# an electronic instrument in real time.
#
# MIDI does not represent musical sound directly, but represents performance
# information encoding instructions about how music is to be produced.

# %% [markdown]
# ## MIDI Representation
#
# The **Standard MIDI File (SMF)** specification describes how MIDI data should
# be stored on a computer. A MIDI file contains:
# - List of MIDI messages with timestamps
# - Meta messages for software processing
#
# Key MIDI messages:
# - **Note-on/Note-off**: Start and end of notes
# - **MIDI note number**: Integer 0-127 encoding pitch (C0 to G#9)
# - **Key velocity**: Integer 0-127 controlling intensity
# - **MIDI channel**: Integer 0-15 specifying instrument
# - **Timestamp**: Ticks to wait before executing command

# %% [markdown]
# ## Timing Information in MIDI
#
# MIDI can handle both musical and physical timing:
# - **PPQN**: Pulses per quarter note (e.g., 120)
# - **Tempo messages**: Microseconds per quarter note
# - **BPM**: Beats per minute (derived from tempo)

# %%
import os
import sys
from matplotlib import pyplot as plt
import pretty_midi
import pandas as pd
import IPython.display as ipd

sys.path.append('..')
import libfmp.c1

# %%
# Read and parse MIDI file
fn = os.path.join('..', 'data', 'C1', 'FMP_C1_F13a_Beethoven_FateMotive_Sibelius-Tracks.mid')
midi_data = pretty_midi.PrettyMIDI(fn)
midi_list = []

for instrument in midi_data.instruments:
    for note in instrument.notes:
        start = note.start
        end = note.end
        pitch = note.pitch
        velocity = note.velocity
        midi_list.append([start, end, pitch, velocity, instrument.name])

midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))

df = pd.DataFrame(midi_list, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
html = df.to_html(index=False)
ipd.HTML(html)

# %% [markdown]
# ## Synthesizing MIDI
#
# PrettyMIDI can synthesize MIDI data with sinusoidal sounds.

# %%
Fs = 22050
audio_data = midi_data.synthesize(fs=Fs)
ipd.Audio(audio_data, rate=Fs)

# %% [markdown]
# ## Piano Roll Visualization from MIDI

# %%
def midi_to_list(midi):
    """Convert a midi file to a list of note events

    Args:
        midi (str or pretty_midi.PrettyMIDI): Path to midi file or PrettyMIDI object

    Returns:
        score (list): List of note events [start, duration, pitch, velocity, label]
    """
    if isinstance(midi, str):
        midi_data = pretty_midi.pretty_midi.PrettyMIDI(midi)
    elif isinstance(midi, pretty_midi.pretty_midi.PrettyMIDI):
        midi_data = midi
    else:
        raise RuntimeError('midi must be a path to a midi file or pretty_midi.PrettyMIDI')

    score = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            duration = note.end - start
            pitch = note.pitch
            velocity = note.velocity / 127.  # normalize to [0, 1]
            score.append([start, duration, pitch, velocity, instrument.name])
    return score


# %%
score = midi_to_list(midi_data)
libfmp.c1.visualize_piano_roll(score, figsize=(8, 3), velocity_alpha=True)
plt.show()

# %% [markdown]
# ## Bach Fugue BWV 846 from MIDI
#
# The four voices (soprano, alto, tenor, basso) are encoded by four different
# MIDI channels.

# %%
fn = os.path.join('..', 'data', 'C1', 'FMP_C1_F12_Bach_BWV846_Sibelius-Tracks.mid')
midi_data = pretty_midi.PrettyMIDI(fn)
score = midi_to_list(midi_data)
libfmp.c1.visualize_piano_roll(score, figsize=(8, 3), velocity_alpha=True)
plt.show()

# %% [markdown]
# ## Conversion from MIDI to CSV
#
# We can convert MIDI to CSV format using pandas.

# %%
fn_in = os.path.join('..', 'data', 'C1', 'FMP_C1_F12_Bach_BWV846_Sibelius-Tracks.mid')
fn_out = os.path.join('..', 'output', 'C1', 'FMP_C1_F12_Bach_BWV846_Sibelius-Tracks.csv')
midi_data = pretty_midi.PrettyMIDI(fn_in)
score = midi_to_list(midi_data)
df = pd.DataFrame(score, columns=['Start', 'Duration', 'Pitch', 'Velocity', 'Instrument'])
df.to_csv(fn_out, sep=';', quoting=2, float_format='%.3f', index=False)

print('Score as list:')
print(score[0:3])
print('\nScore as pandas DataFrame')
print(df.loc[0:2, :])

# %% [markdown]
# ## Using libfmp Functions

# %%
fn = os.path.join('..', 'data', 'C1', 'FMP_C1_F13a_Beethoven_FateMotive_Sibelius.mid')
fn_out = os.path.join('..', 'output', 'C1', 'FMP_C1_F13a_Beethoven_FateMotive_Sibelius.csv')

score = libfmp.c1.midi_to_list(fn)
libfmp.c1.visualize_piano_roll(score, figsize=(8, 3))
plt.show()
libfmp.c1.list_to_csv(score, fn_out)

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Frank Zalkow and Meinard Muller.

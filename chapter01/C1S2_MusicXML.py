# %% [markdown]
# # Symbolic Format: MusicXML
#
# Following Section 1.2.3 of [Muller, FMP, Springer 2015], we look at a symbolic
# music representation called MusicXML.

# %% [markdown]
# ## Score Representations
#
# **Score representations** yield explicit information about musical symbols such
# as staff systems, clefs, time signatures, notes, rests, accidentals, and dynamics.
# Unlike MIDI, score representations are much closer to what is shown in sheet music.
# For example, D#4 and Eb4 would be distinguishable.

# %% [markdown]
# ## MusicXML
#
# **MusicXML** is a universal format for storing and sharing music files between
# notation applications. It follows the XML paradigm - a textual format that is
# both human and machine readable.
#
# A note element in MusicXML contains:
# - `<pitch>`: pitch class, alter, octave
# - `<duration>`: duration in quarter notes
# - `<type>`: how the note is depicted

# %%
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import IPython.display as ipd
import music21 as m21

sys.path.append('..')
import libfmp.c1

# %%
# Read and display MusicXML note element
fn = os.path.join('..', 'data', 'C1', 'FMP_C1_F15_Eflat.xml')

with open(fn, 'r') as stream:
    xml_str = stream.read()

start = xml_str.find('<note')
end = xml_str[start:].find('</note>') + start + len('</note>')
print(xml_str[start:end])

# %% [markdown]
# ## Processing MusicXML with music21
#
# The Python package `music21` is a toolkit for computer-aided musicology that
# can read MusicXML files.

# %%
fn = os.path.join('..', 'data', 'C1', 'FMP_C1_F01_Beethoven_FateMotive_Sibelius.xml')


def xml_to_list(xml):
    """Convert a music xml file to a list of note events

    Args:
        xml (str or music21.stream.Score): Path to xml file or music21 Score

    Returns:
        score (list): List of note events [start, duration, pitch, velocity, label]
    """
    if isinstance(xml, str):
        xml_data = m21.converter.parse(xml)
    elif isinstance(xml, m21.stream.Score):
        xml_data = xml
    else:
        raise RuntimeError('xml must be a path to a xml file or music21.stream.Score')

    score = []
    for part in xml_data.parts:
        instrument = part.getInstrument().instrumentName

        for note in part.flatten().notes:
            if note.isChord:
                start = note.offset
                duration = note.quarterLength
                for chord_note in note.pitches:
                    pitch = chord_note.ps
                    volume = note.volume.realized
                    score.append([start, duration, pitch, volume, instrument])
            else:
                start = note.offset
                duration = note.quarterLength
                pitch = note.pitch.ps
                volume = note.volume.realized
                score.append([start, duration, pitch, volume, instrument])

    score = sorted(score, key=lambda x: (x[0], x[2]))
    return score


# %%
xml_data = m21.converter.parse(fn)
xml_list = xml_to_list(xml_data)

df = pd.DataFrame(xml_list[:9], columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
html = df.to_html(index=False, float_format='%.2f', max_rows=8)
ipd.HTML(html)

# %% [markdown]
# ## Piano Roll Visualization from MusicXML

# %%
libfmp.c1.visualize_piano_roll(xml_list, figsize=(8, 3), velocity_alpha=True,
                               xlabel='Time (quarter lengths)')
plt.show()

# %% [markdown]
# ## Orchestral Score Visualization

# %%
fn = os.path.join('..', 'data', 'C1', 'FMP_C1_F10_Beethoven_Fifth-MM1-21_Sibelius-Orchestra.xml')
xml_data = m21.converter.parse(fn)
xml_list = xml_to_list(xml_data)
libfmp.c1.visualize_piano_roll(xml_list, figsize=(10, 7), velocity_alpha=False,
                               colors='gist_rainbow', xlabel='Time (quarter lengths)')
plt.show()

# %% [markdown]
# ## Conversion from MusicXML to CSV

# %%
fn = os.path.join('..', 'data', 'C1', 'FMP_C1_F10_Beethoven_Fifth-MM1-21_Sibelius-Orchestra.xml')
fn_out = os.path.join('..', 'output', 'C1', 'FMP_C1_F10_Beethoven_Fifth-MM1-21_Sibelius-Orchestra.csv')
xml_data = m21.converter.parse(fn)
xml_list = xml_to_list(xml_data)
df = pd.DataFrame(xml_list, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
df.to_csv(fn_out, sep=';', quoting=2, float_format='%.3f')

print('Score as list:')
print(xml_list[0:3])
print('\nScore as pandas DataFrame')
print(df.loc[0:2, :])

# %% [markdown]
# ## Using libfmp Functions

# %%
fn = os.path.join('..', 'data', 'C1', 'FMP_C1_F13a_Beethoven_FateMotive_Sibelius.xml')
fn_out = os.path.join('..', 'output', 'C1', 'FMP_C1_F13a_Beethoven_FateMotive_Sibelius.csv')

score = libfmp.c1.xml_to_list(fn)
libfmp.c1.visualize_piano_roll(xml_list, figsize=(10, 7), velocity_alpha=True,
                               colors='gist_rainbow', xlabel='Time (quarter lengths)')
plt.show()
libfmp.c1.list_to_csv(score, fn_out)

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Frank Zalkow and Meinard Muller.

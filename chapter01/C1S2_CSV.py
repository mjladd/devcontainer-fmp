# %% [markdown]
# # Symbolic Format: CSV
#
# Following Section 1.2 of [Muller, FMP, Springer 2015], we introduce a simple
# symbolic music representation encoded in comma-separated values (CSV) format.

# %% [markdown]
# ## CSV Format
#
# A piano-roll representation yields a visualization of note events, where each
# note is encoded by start, duration, and pitch parameters. In our CSV format,
# each note event is a line with five parameters:
# - `start`: onset time (seconds or measures)
# - `duration`: note length (seconds or measures)
# - `pitch`: MIDI note number
# - `velocity`: intensity (0.0 to 1.0)
# - `label`: instrument, voice, or staff information

# %%
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import pandas as pd
import IPython.display as ipd

%matplotlib inline

# %%
# Read and display CSV file
fn = os.path.join('..', 'data', 'C1', 'FMP_C1_F01_Beethoven_FateMotive_Sibelius.csv')

with open(fn, 'r', encoding='utf-8') as file:
    csv_str = file.read()

print(csv_str)

# %% [markdown]
# ## Python Library `pandas`
#
# The Python library `pandas` provides easy-to-use data structures and data
# analysis tools. It can read CSV files into a `DataFrame` - a two-dimensional
# tabular data structure with labeled axes.

# %%
df = pd.read_csv(fn, sep=';')
print(df.loc[0:5, :])

# %%
# Render DataFrame as HTML table
html = df.loc[0:5, :].to_html(index=False)
ipd.HTML(html)

# %% [markdown]
# ## Piano-Roll Visualization
#
# The note events stored in the CSV file can be visualized using a piano-roll
# representation. Rectangle colors indicate different labels associated with
# note events.

# %%
import sys
sys.path.append('..')
import libfmp.b


def csv_to_list(csv):
    """Convert a csv score file to a list of note events

    Args:
        csv (str or pd.DataFrame): Either a path to a csv file or a data frame

    Returns:
        score (list): A list of note events where each note is specified as
            [start, duration, pitch, velocity, label]
    """
    if isinstance(csv, str):
        df = libfmp.b.read_csv(csv)
    elif isinstance(csv, pd.DataFrame):
        df = csv
    else:
        raise RuntimeError('csv must be a path to a csv file or pd.DataFrame')

    score = []
    for i, (start, duration, pitch, velocity, label) in df.iterrows():
        score.append([start, duration, pitch, velocity, label])
    return score


def visualize_piano_roll(score, xlabel='Time (seconds)', ylabel='Pitch', colors='FMP_1',
                         velocity_alpha=False, figsize=(12, 4), ax=None, dpi=72):
    """Plot a piano roll visualization

    Args:
        score: List of note events
        xlabel: Label for x axis (Default value = 'Time (seconds)')
        ylabel: Label for y axis (Default value = 'Pitch')
        colors: Color specification (Default value = 'FMP_1')
        velocity_alpha: Use velocity for alpha value (Default value = False)
        figsize: Width, height in inches (Default value = (12, 4))
        ax: The Axes instance to plot on (Default value = None)
        dpi: Dots per inch (Default value = 72)

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot(1, 1, 1)

    labels_set = sorted(set([note[4] for note in score]))
    colors = libfmp.b.color_argument_to_dict(colors, labels_set)

    pitch_min = min(note[2] for note in score)
    pitch_max = max(note[2] for note in score)
    time_min = min(note[0] for note in score)
    time_max = max(note[0] + note[1] for note in score)

    for start, duration, pitch, velocity, label in score:
        if velocity_alpha is False:
            velocity = None
        rect = patches.Rectangle((start, pitch - 0.5), duration, 1, linewidth=1,
                                  edgecolor='k', facecolor=colors[label], alpha=velocity)
        ax.add_patch(rect)

    ax.set_ylim([pitch_min - 1.5, pitch_max + 1.5])
    ax.set_xlim([min(time_min, 0), time_max + 0.5])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend([patches.Patch(linewidth=1, edgecolor='k', facecolor=colors[key]) for key in labels_set],
              labels_set, loc='upper right', framealpha=1)

    if fig is not None:
        plt.tight_layout()

    return fig, ax


# %%
# Visualize Beethoven's Fate Motive
score = csv_to_list(df)
visualize_piano_roll(score, colors=['red', 'blue'], figsize=(8, 3))
plt.show()

# %% [markdown]
# ## Piano-Roll for Orchestral Version
#
# Piano-roll representation for a full orchestral score of the first 21 measures
# of Beethoven's Fifth.

# %%
fn = os.path.join('..', 'data', 'C1', 'FMP_C1_F10_Beethoven_Fifth-MM1-21_Sibelius-Orchestra.csv')

df = pd.read_csv(fn, sep=';')
score_list = csv_to_list(df)
visualize_piano_roll(score_list, figsize=(10, 7), colors='gist_rainbow')
plt.show()

# %% [markdown]
# ## Piano Roll for Fugue
#
# Piano-roll representation of Bach's four-voice Fugue BWV 846 in C major.
# The four voices are labeled: Soprano, Alto, Tenor, and Basso.

# %%
fn = os.path.join('..', 'data', 'C1', 'FMP_C1_F12_Bach_BWV846_Sibelius-Tracks.csv')

df = pd.read_csv(fn, sep=';')
score_list = csv_to_list(df)
visualize_piano_roll(score_list, figsize=(8, 3))
plt.show()

# %% [markdown]
# ## Using libfmp Functions
#
# The CSV conversion and piano roll visualization functions are available in libfmp.

# %%
import libfmp.c1

fn = os.path.join('..', 'data', 'C1', 'FMP_C1_F01_Beethoven_FateMotive_Sibelius.csv')
score = libfmp.c1.csv_to_list(fn)
libfmp.c1.visualize_piano_roll(score, figsize=(8, 3))
plt.show()

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Frank Zalkow and Meinard Muller.

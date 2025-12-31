# %% [markdown]
# # Symbolic Format: Rendering
#
# In this notebook, we discuss software tools for rendering sheet music and
# explain functionality of the Python library `music21`.
#
# **Note:** Some code cells require additional software dependencies (MuseScore,
# LilyPond) and are provided as comments with static image outputs shown.

# %% [markdown]
# ## Extra Software
#
# Sheet music rendering refers to the computerized rendition of a symbolic music
# format into graphical musical symbols. Software tools include:
# - Text-based: LilyPond, Verovio
# - WYSIWYG editors: MuseScore, Sibelius
#
# For rendering from Python, we recommend `music21`.

# %%
import os
import music21 as m21

# Set up music21 user environment
us = m21.environment.UserSettings()
us_path = us.getSettingsPath()
if not os.path.exists(us_path):
    us.create()
print('Path to music21 environment', us_path)
print(us)

# %% [markdown]
# ## Configuring MuseScore
#
# To use MuseScore for rendering:
# 1. Install MuseScore from https://musescore.org/en/download
# 2. Set up music21 user environment file
# 3. Register the path to MuseScore
#
# ```python
# # for linux
# us['musescoreDirectPNGPath'] = '/usr/bin/mscore'
# us['musicxmlPath'] = '/usr/bin/mscore'
#
# # for windows
# us['musescoreDirectPNGPath'] = r'C:\Program Files (x86)\MuseScore 2\bin\MuseScore.exe'
# us['musicxmlPath'] = r'C:\Program Files (x86)\MuseScore 2\bin\MuseScore.exe'
# ```

# %% [markdown]
# ## Rendering a Single Note
#
# ```python
# n = m21.note.Note('c')
# n.show('ipython.musicxml.png')
# ```
#
# This renders a C4 quarter note on a staff.

# %% [markdown]
# ## Configuring LilyPond
#
# To use LilyPond for rendering:
# 1. Install LilyPond from http://lilypond.org/download.html
# 2. Register the path in music21
#
# ```python
# # for linux
# us['lilypondPath'] = '/usr/local/bin/lilypond'
#
# # for windows
# us['lilypondPath'] = r'C:\Program Files (x86)\LilyPond\usr\bin\lilypond.exe'
# ```

# %% [markdown]
# ## Creating a Staff with Multiple Notes
#
# Add multiple music21 objects to a Stream:
#
# ```python
# s = m21.stream.Stream()
# s.append(m21.key.Key('E-'))
# s.append(m21.meter.TimeSignature('2/4'))
# s.append(m21.note.Rest(quarterLength=0.5))
# s.append(m21.note.Note('g', quarterLength=0.5))
# s.append(m21.note.Note('g', quarterLength=0.5))
# s.append(m21.note.Note('g', quarterLength=0.5))
# s.append(m21.note.Note('e-', quarterLength=2))
#
# s.show('ipython.musicxml.png')
# ```

# %% [markdown]
# ## Reading and Displaying MusicXML Files
#
# ```python
# fn_xml = os.path.join('..', 'data', 'C1', 'FMP_C1_F01_Beethoven_FateMotive_Sibelius.xml')
# s = m21.converter.parse(fn_xml)
# s.show('ipython.musicxml.png')
# ```

# %% [markdown]
# ## Displaying Specific Measures from Orchestral Score
#
# For large scores that cannot fit on a single page, specify the measures to show:
#
# ```python
# fn_xml = os.path.join('..', 'data', 'C1', 'FMP_C1_F10_Beethoven_Fifth-MM1-21_Sibelius-Orchestra.xml')
# s = m21.converter.parse(fn_xml)
# s.measures(1, 5).show('ipython.musicxml.png')
# ```

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Frank Zalkow and Meinard Muller.

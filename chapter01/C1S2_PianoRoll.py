# %% [markdown]
# # Piano Roll Representations
#
# Following Section 1.2.1 of [Muller, FMP, Springer 2015], we look at a symbolic
# music representation often referred to as piano roll representation.

# %% [markdown]
# ## Player Pianos
#
# In the late 19th and early 20th century, self-playing **player pianos** became
# quite popular. They contained pneumatic mechanisms to automatically operate key
# and pedal movements according to instructions specified by a prestored piano-roll
# medium.
#
# A **piano roll** is a continuous roll of paper with perforations (holes) punched
# into it. The perforations represent note control data. Rolls for player pianos
# were generally made from recorded performances of musicians, preserving the
# playing of famous pianists including Gustav Mahler, Edvard Grieg, Scott Joplin,
# and George Gershwin.

# %%
import IPython.display as ipd

# Example: Player piano playing "The Entertainer"
print("Player piano example (YouTube video):")
ipd.display(ipd.YouTubeVideo('aseMAEctM1s'))

# %% [markdown]
# ## Piano-Roll Representation
#
# A **piano-roll representation** is a geometric visualization of note information:
# - Horizontal axis: encodes time
# - Vertical axis: encodes pitch
# - Each note is an axis-parallel rectangle encoding:
#   - Onset time (leftmost horizontal coordinate)
#   - Pitch (lower vertical coordinate)
#   - Duration (width of rectangle)

# %% [markdown]
# ## Example: Bach's Fugue BWV 846
#
# The piano-roll representation of Bach's four-voice Fugue BWV 846 in C major
# shows how this visualization clearly displays the different voices and the
# occurrences of the fugue theme.

# %%
# Example: Animated piano-roll representation for Bach's Fugue BWV 578
print("Bach's Fugue BWV 578 - animated piano roll (YouTube video):")
ipd.display(ipd.YouTubeVideo('ddbxFi3-UO4', start=1))

# %% [markdown]
# ## Advantages of Piano-Roll Representation
#
# While piano-roll representations are a considerable simplification of sheet
# music notation, they visually describe the most important attributes of musical
# notes in an easy-to-understand way. They serve as a **mid-level representation**
# for establishing semantic relations across various manifestations of music.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller.

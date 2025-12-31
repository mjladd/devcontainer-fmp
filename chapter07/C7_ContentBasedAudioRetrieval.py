# %% [markdown]
# # Content-Based Audio Retrieval
#
# Following Chapter 7 of [Müller, FMP, Springer 2015], we give in this notebook an introduction
# to content-based audio retrieval. In particular, as described in Section 7.3, we discuss a
# retrieval scenario referred to as version identification.

# %% [markdown]
# ## Introduction
#
# The revolution in music distribution and storage brought about by digital technology has
# fueled tremendous interest in and attention to the ways that information technology can be
# applied to this kind of content. The general field of **information retrieval** (IR) is
# devoted to the task of organizing information and of making it accessible and useful. An
# information retrieval process begins when a user specifies his/her information needs by
# means of a **query**. The retrieval system should then deliver from a given data collection
# all **documents** or **items** that are somehow related to the query.
#
# Most of the available services for music recommendation and playlist generation rely on
# metadata and textual annotations of the actual audio content. While **text-based** retrieval
# systems can be very powerful, they require the audio material to be enriched with suitable
# metadata—an assumption that is often not valid, in particular for less popular music or
# music material that is scattered in unstructured data collections. To handle such scenarios,
# one requires **content-based retrieval** systems that only make use of the raw music data,
# rather than relying on manually generated metadata.

# %% [markdown]
# ## Query-By-Example
#
# Many content-based retrieval strategies follow the **query-by-example** paradigm: given a
# music representation or a fragment of it (used as a query or example), the task is to
# automatically retrieve documents from a music collection containing parts or aspects that
# are similar to the query.
#
# * **Audio identification** (sometimes also called **audio fingerprinting**): Given a small
#   audio fragment as query, the task consists in identifying the particular audio recording
#   that is the source of the query.
#
# * **Audio matching**: Given a query fragment, the goal is to retrieve all audio excerpts
#   that musically correspond to the query. One explicitly allows semantically motivated
#   variations as they typically occur in different performances and arrangements.
#
# * **Version identification** (sometimes also called **cover song retrieval**): Deals not
#   only with performance variations in instrumentation and tempo, but also with more extreme
#   variations concerning the musical structure, key, or melody.
#
# * **Category-based** retrieval scenarios (including **genre classification**): The
#   similarity relationships are rather vague and express cultural or musicological categories.

# %% [markdown]
# ## Specificity and Granularity
#
# Content-based retrieval strategies can be loosely classified according to their
# **specificity** and **granularity**.
#
# * The **specificity** of a retrieval system refers to the degree of similarity between the
#   query and the database documents to be retrieved. Highly specific retrieval systems return
#   exact or near copies of the query, whereas low-specific retrieval systems return
#   semantically related matches that may be quite different from the original query.
#
# * The **granularity** refers to the temporal level considered in the retrieval scenario. In
#   **fragment-level** retrieval scenarios, the query consists of a short fragment of an audio
#   recording, and the goal is to retrieve all related fragments. In **document-level**
#   retrieval, the query reflects characteristics of an entire document and is compared with
#   entire documents of the database.

# %% [markdown]
# ## Versions in Music
#
# In Western culture, when speaking of a **piece of music**, one typically thinks of a
# specific composition given in music notation or given in the form of a recorded track.
# Instead of trying to give a formal definition of "version", let us consider some typical
# examples:
#
# * An **arrangement** refers to a reworking of a piece of music so that it can be played by
#   instruments different from the ones notated in the original score.
# * A **piano transcription** is an arrangement of symphonic and chamber music so that it can
#   be played on one or two pianos.
# * A **cover version** or **cover song** loosely refers to a new performance of a previously
#   released song by someone other than the original artist.
# * A **remix** is a recording that has been edited or completely recreated to sound different
#   from the original version.
# * **Sampling** refers to the technique of taking portions of one recording and reusing them
#   as a "new" instrument in a different piece.
#
# A version may differ from the original recording in many ways, possibly including
# significant changes in timbre, instrumentation, tempo, key, harmony, melody, lyrics, and
# musical structure.

# %%
import IPython.display as ipd
import numpy as np

# %% [markdown]
# ## Further Notes
#
# In the subsequent notebooks of this chapter, we cover three content-based audio retrieval
# scenarios:
#
# * Audio identification: Fingerprinting techniques
# * Audio matching: Subsequence dynamic time warping techniques
# * Version identification: Common subsequence matching techniques

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

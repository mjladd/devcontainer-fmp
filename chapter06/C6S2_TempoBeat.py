# %% [markdown]
# # Tempo and Beat
#
# Following Section 6.2.1 of [Müller, FMP, Springer 2015], we introduce in this notebook
# the concepts of beat and tempo.

# %% [markdown]
# ## Basic Notions and Assumptions
#
# Temporal and structural regularities are perhaps the most important incentives for people
# to get involved and to interact with music. It is the **beat** that drives music forward
# and provides the temporal framework of a piece of music. Intuitively, the beat corresponds
# to the pulse a human taps along when listening to music. The beat is often described as a
# sequence of perceived pulse positions, which are typically equally spaced in time and
# specified by two parameters: the **phase** and the **period**. The term **tempo** refers
# to the rate of the pulse and is given by the reciprocal of the beat period.
#
# The extraction of tempo and beat information from audio recordings is a challenging problem
# in particular for music with weak note onsets and local tempo changes. For example, in the
# case of romantic piano music, the pianist often takes the freedom of speeding up and slowing
# down the tempo—an artistic means also referred to as **tempo rubato**. There is a wide range
# of music where the notions of tempo and beat remain rather vague or are even nonexistent.
# Sometimes, the rhythmic flow of music is deliberately interrupted or disturbed by
# **syncopation**, where certain notes outside the regular grid of beat positions are stressed.
#
# To make the problem of tempo and beat tracking feasible, most automated approaches rely on
# two basic assumptions:
#
# * The first assumption is that beat positions occur at note onset positions.
# * The second assumption is that beat positions are more or less equally spaced—at least
#   for a certain period of time.
#
# Even though both assumptions may be violated and inappropriate for certain types of music,
# they are convenient and reasonable for a wide range of music including most rock and popular songs.

# %%
import os, sys
import sys
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import librosa
import IPython.display as ipd
import pandas as pd
sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c6
%matplotlib inline

def plot_sonify_novelty_beats(fn_wav, fn_ann, title=''):
    ann, label_keys = libfmp.c6.read_annotation_pos(fn_ann, label='onset', header=0)
    df = pd.read_csv(fn_ann, sep=';', keep_default_na=False, header=None)
    beats_sec = df.values
    Fs = 22050
    x, Fs = librosa.load(fn_wav, Fs)
    x_duration = len(x)/Fs
    nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=256, gamma=1, M=10, norm=1)
    figsize=(8,1.5)
    fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, color='k', figsize=figsize,
                title=title)
    libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                        nontime_axis=True, time_min=0, time_max=x_duration)
    plt.show()
    x_beats = librosa.clicks(beats_sec, sr=Fs, click_freq=1000, length=len(x))
    ipd.display(ipd.Audio(x + x_beats, rate=Fs))

title = 'Borodin: String Quartet No. 2, 3rd movement'
fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_Borodin-sec39_RWC_quarter.csv')
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_Borodin-sec39_RWC.wav')
plot_sonify_novelty_beats(fn_wav, fn_ann, title)

title = 'Chopin: Op.68, No. 3'
fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_Chopin.csv')
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_Chopin.wav')
plot_sonify_novelty_beats(fn_wav, fn_ann, title)

title = 'Fauré: Op.15'
fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_Faure_Op015-01-sec0-12_SMD126.csv')
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_Faure_Op015-01-sec0-12_SMD126.wav')
plot_sonify_novelty_beats(fn_wav, fn_ann, title)

# %% [markdown]
# ## Tempogram Representations
#
# In Fourier analysis, a (magnitude) spectrogram is a time–frequency representation of a given
# signal. A large value $\mathcal{Y}(t,\omega)$ of a spectrogram indicates that the signal
# contains at time instance $t$ a periodic component that corresponds to the frequency $\omega$.
# We now introduce a similar concept referred to as a **tempogram**, which indicates for each
# time instance the local relevance of a specific tempo for a given music recording.
# Mathematically, we model a tempogram as a function
#
# $$\mathcal{T}:\mathbb{R}\times \mathbb{R}_{>0}\to \mathbb{R}_{\geq 0}$$
#
# depending on a time parameter $t\in\mathbb{R}$ measured in seconds and a tempo parameter
# $\tau \in \mathbb{R}_{>0}$ measured in **beats per minute** (**BPM**). Intuitively, the value
# $\mathcal{T}(t,\tau)$ indicates the extent to which the signal contains a locally periodic
# pulse of a given tempo $\tau$ in a neighborhood of time instance $t$.
#
# Most approaches for deriving a tempogram representation from a given audio recording proceed
# in two steps:
#
# * Based on the assumption that pulse positions usually go along with note onsets, the music
#   signal is first converted into a novelty function.
# * In the second step, the locally periodic behavior of the novelty function is analyzed.

# %% [markdown]
# ## Pulse Levels
#
# One major problem in determining the tempo of a music recording arises from the fact that
# pulses in music are often organized in complex hierarchies that represent the rhythm. In
# particular, there are various levels that are presumed to contribute to the human perception
# of tempo and beat:
#
# * The **tactus** level typically corresponds to the quarter note level and often matches the
#   foot tapping rate.
# * Thinking at a larger musical scale, one may also perceive the tempo at the **measure** level,
#   in particular when listening to fast music or to highly expressive music with strong rubato.
# * Finally, one may also consider the **tatum** (temporal atom) level, which refers to the
#   fastest repetition rate of musically meaningful accents occurring in the signal.

# %%
title = 'Pulse on measure level'
fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_HappyBirthday_measure.csv')
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_HappyBirthday.wav')
plot_sonify_novelty_beats(fn_wav, fn_ann, title)

title = 'Pulse on tactus level'
fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_HappyBirthday_tactus.csv')
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_HappyBirthday.wav')
plot_sonify_novelty_beats(fn_wav, fn_ann, title)

title = 'Pulse on tatum level'
fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_HappyBirthday_tatum.csv')
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_HappyBirthday.wav')
plot_sonify_novelty_beats(fn_wav, fn_ann, title)

# %% [markdown]
# ## Tempo Octave, Harmonic, and Subharmonic
#
# Often the tempo ambiguity that arises from the existence of different pulse levels is also
# reflected in a tempogram $\mathcal{T}$. Higher pulse levels often correspond to integer
# multiples $\tau,2\tau,3\tau,\ldots$ of a given tempo $\tau$. As with pitch, we call such
# integer multiples **(tempo) harmonics** of $\tau$. Furthermore, integer fractions
# $\tau,\tau/2,\tau/3,\ldots$ are referred to as **(tempo) subharmonics** of $\tau$. Analogous
# to the notion of an octave for musical pitches, the difference between two tempi with half
# or double the value is called a **tempo octave**.

# %% [markdown]
# ## Global Tempo
#
# Assuming a more or less steady tempo, it suffices to determine one **global** tempo value for
# the entire recording. Such a value may be obtained by averaging the tempo values obtained from
# a frame-wise periodicity analysis. For example, based on a tempogram representation, one can
# average the tempo values over all time frames to obtain a function
# $\mathcal{T}_\mathrm{Average}:\Theta\to\mathbb{R}_{\geq 0}$ that only depends on $\tau\in\Theta$.
#
# The maximum of this function then yields an estimate for the global tempo of the recording.
# Of course, more refined methods for estimating a single tempo value may be applied. For
# example, instead of using a simple average, one may apply median filtering, which is more
# robust to outliers and noise.

# %% [markdown]
# ## Further Notes
#
# In the following notebooks, we introduce two conceptually different methods for computing
# tempogram representations:
#
# * Fourier-based tempograms
# * Autocorrelation-based tempograms

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Vlora Arifi-Müller.

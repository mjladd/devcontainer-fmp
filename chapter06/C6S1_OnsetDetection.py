# %% [markdown]
# # Onset Detection
#
# Following Section 6.1 of [Müller, FMP, Springer 2015], we introduce in this notebook
# the task referred to as onset detection.

# %% [markdown]
# ## Musical Onsets
#
# The notion of a musical onset can be rather vague and is related to other concepts
# such as attacks or transients:
#
# * **Attack**: The phase where the sound builds up with a sharply increasing amplitude envelope
# * **Transient**: A noise-like sound component of short duration and high amplitude
# * **Onset**: The single instant that marks the beginning of the transient

# %% [markdown]
# ## General Pipeline
#
# Many approaches for onset detection follow a similar algorithmic pipeline:
#
# 1. Convert the signal into a suitable feature representation
# 2. Apply a derivative operator to derive a novelty function
# 3. Apply a peak-picking algorithm to locate onset candidates
#
# We study four different approaches for computing novelty functions:
# * Energy-based novelty
# * Spectral-based novelty
# * Phase-based novelty
# * Complex-domain novelty

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

def plot_wav_spectrogram(fn_wav, xlim=None, audio=True):
    """Plot waveform and computed spectrogram and may display audio
    Notebook: C6/C6S1_OnsetDetection.ipynb
    """
    Fs = 22050
    x, Fs = librosa.load(fn_wav, Fs)
    plt.figure(figsize=(8,2))
    ax = plt.subplot(1,2,1)
    libfmp.b.plot_signal(x, Fs, ax=ax)
    if xlim!=None: plt.xlim(xlim)
    ax = plt.subplot(1,2,2)
    N, H = 512, 256
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Y = np.log(1 + 10 * np.abs(X))
    libfmp.b.plot_matrix(Y, Fs=Fs/H, Fs_F=N/Fs, ax=[ax], colorbar=False)
    plt.ylim([0,5000])
    if xlim is not None: plt.xlim(xlim)
    plt.tight_layout()
    plt.show()
    if audio: ipd.display(ipd.Audio(x, rate=Fs))


fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F04_Impulse.wav')
plot_wav_spectrogram(fn_wav)

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F04_NoteC4_Piano.wav')
plot_wav_spectrogram(fn_wav)

# %% [markdown]
# ## Soft Onsets
#
# When there is no clear attack phase, such as for nonpercussive music with soft onsets
# and blurred note transitions, the detection of onsets is much more challenging.

# %%
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F04_NoteC4_Violin.wav')
plot_wav_spectrogram(fn_wav)

# %% [markdown]
# ## Polyphonic Music
#
# The detection of individual note onsets becomes even harder when dealing with complex
# polyphonic music due to masking effects.

# %%
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_Borodin-sec39_RWC.wav')
plot_wav_spectrogram(fn_wav)
print('Plot of the first six seconds:')
plot_wav_spectrogram(fn_wav, xlim=[0,6], audio=False)

# %% [markdown]
# ## Example: Another One Bites the Dust
#
# We consider an excerpt of "Another one bites the dust" by Queen. Starting with an
# offbeat consisting of two sixteenth notes played only by bass, four percussive beats
# follow. Furthermore, between each two subsequent beats, there is an additional hihat stroke.

# %%
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F01_Queen.wav')
plot_wav_spectrogram(fn_wav, audio=False)

# %% [markdown]
# ## Annotation Reading Function

# %%
def read_annotation_pos(fn_ann, label='', header=True, print_table=False):
    """Read and convert file containing either list of pairs (number,label) or list of (number)

    Notebook: C6/C6S1_OnsetDetection.ipynb

    Args:
        fn_ann (str): Name of file
        label (str): Name of label (Default value = '')
        header (bool): Assumes header (True) or not (False) (Default value = True)
        print_table (bool): Prints table if True (Default value = False)

    Returns:
        ann (list): List of annotations
        label_keys (dict): Dictionaries specifying color and line style used for labels
    """
    df = libfmp.b.read_csv(fn_ann, header=header)
    if print_table:
        print(df)
    num_col = df.values[0].shape[0]
    if num_col == 1:
        df = df.assign(label=[label] * len(df.index))
    ann = df.values.tolist()

    label_keys = {'beat': {'linewidth': 2, 'linestyle': ':', 'color': 'r'},
                  'onset': {'linewidth': 1, 'linestyle': ':', 'color': 'r'}}
    return ann, label_keys

fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_F01_Queen.csv')
ann, label_keys = read_annotation_pos(fn_ann, print_table=True)

x, Fs = librosa.load(fn_wav)
x_duration = len(x)/Fs
nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs)

fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 3))
libfmp.b.plot_signal(nov, Fs_nov, ax=ax[0], color='k', title='Novelty function');
libfmp.b.plot_annotation_line(ann, ax=ax[1], label_keys=label_keys,
                    time_min=0, time_max=x_duration)
ax[1].set_title('Annotated onset and beat positions')
ax[1].set_xlabel('Time (seconds)')

plt.tight_layout()

# %% [markdown]
# ## Peak Picking
#
# We apply a peak picking strategy to locate the local maxima of the novelty function.
# The positions of the peaks are our candidates for onset positions.

# %%
peaks, properties = signal.find_peaks(nov, prominence=0.02)
T_coef = np.arange(nov.shape[0]) / Fs_nov
peaks_sec = T_coef[peaks]
fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, color='k',
                    title='Novelty function with detected peaks')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                    nontime_axis=True, time_min=0, time_max=x_duration)
plt.plot(peaks_sec, nov[peaks], 'ro')
plt.show()

x_peaks = librosa.clicks(peaks_sec, sr=Fs, click_freq=1000, length=len(x))
ipd.display(ipd.Audio(x + x_peaks, rate=Fs))

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

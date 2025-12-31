# %% [markdown]
# # Music Structure Analysis: General Principles
#
# Following Section 4.1 of [Müller, FMP, Springer 2015], we discuss in this notebook
# general principles for segmenting and structuring music recordings.

# %% [markdown]
# ## General Principles
#
# Music structure analysis is a multifaceted and often ill-defined problem that depends
# on many different aspects. First of all, the complexity of the problem depends on the
# kind of music representation to be analyzed. Second, there are various principles
# including **homogeneity**, **repetition**, and **novelty** that a segmentation may be
# based on. Third, one also has to account for different musical dimensions, such as
# melody, harmony, rhythm, or timbre. Finally, the segmentation and structure largely
# depend on the musical context and the **temporal hierarchy** to be considered.

# %% [markdown]
# ## Segmentation and Structure Analysis
#
# The tasks of segmenting and structuring multimedia documents are of fundamental
# importance not only for the processing of music signals but also for general
# audio-visual content. **Segmentation** typically refers to the process of partitioning
# a given document into multiple segments with the goal of simplifying the representation
# into something that is more meaningful and easier to analyze than the original document.
#
# Going beyond mere segmentation, the goal of **structure analysis** is to also find and
# understand the relationships between the segments. The challenge in computational music
# structure analysis is that structure in music arises from many different kinds of
# **relationships** including **repetition**, **contrast**, **variation**, and **homogeneity**.
#
# * First, **repetition-based** methods are used to identify recurring patterns.
# * Second, **novelty-based** methods are employed to detect transitions between contrasting parts.
# * Third, **homogeneity-based** methods are used to determine passages that are consistent
#   with respect to some musical property.

# %% [markdown]
# ## Musical Structure
#
# To specify musical structures, we now introduce some terminology. First of all, we
# distinguish between a **piece of music** (in an abstract sense) and a particular
# **audio recording** (an actual performance) of the piece. The term **part** is used
# in the context of the abstract music domain, whereas the term **segment** is used for
# the audio domain. Musical parts are typically denoted by the capital letters A,B,C,...
# in the order of their first occurrence, where numbers indicate the order of repeated
# occurrences.
#
# As an example, we consider the Hungarian Dance No. 5 by Johannes Brahms. The musical
# structure is A1A2B1B2CA3B3B4D, which consists of three repeating A-parts, four
# repeating B-parts, as well as a C-part and a short closing D-part.

# %% [markdown]
# ## Audio Structure Analysis
#
# Given a recording of a piece of music, the goal of **audio structure analysis** (as
# considered in this chapter) is to find the segments within the recording that
# correspond to the various parts of a musical structure.

# %%
import numpy as np
import os, sys, librosa
from scipy import signal
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import IPython.display as ipd
import pandas as pd

sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c3
import libfmp.c4
import libfmp.c6

%matplotlib inline

# %% [markdown]
# ## Structure Annotations

# %%
def convert_structure_annotation(ann, Fs=1, remove_digits=False, index=False):
    """Convert structure annotations

    Notebook: C4/C4S1_MusicStructureGeneral.ipynb

    Args:
        ann (list): Structure annotions
        Fs (scalar): Sampling rate (Default value = 1)
        remove_digits (bool): Remove digits from labels (Default value = False)
        index (bool): Round to nearest integer (Default value = False)

    Returns:
        ann_converted (list): Converted annotation
    """
    ann_converted = []
    for r in ann:
        s = r[0] * Fs
        t = r[1] * Fs
        if index:
            s = int(np.round(s))
            t = int(np.round(t))
        if remove_digits:
            label = ''.join([i for i in r[2] if not i.isdigit()])
        else:
            label = r[2]
        ann_converted = ann_converted + [[s, t, label]]
    return ann_converted


get_color_for_annotation_file = libfmp.c4.get_color_for_annotation_file

def read_structure_annotation(fn_ann, fn_ann_color='', Fs=1, remove_digits=False, index=False):
    """Read and convert structure annotation and colors

    Notebook: C4/C4S1_MusicStructureGeneral.ipynb

    Args:
        fn_ann (str): Path and filename for structure annotions
        fn_ann_color (str): Filename used to identify colors (Default value = '')
        Fs (scalar): Sampling rate (Default value = 1)
        remove_digits (bool): Remove digits from labels (Default value = False)
        index (bool): Round to nearest integer (Default value = False)

    Returns:
        ann (list): Annotations
        color_ann (dict): Color scheme
    """
    df = libfmp.b.read_csv(fn_ann)
    ann = [(start, end, label) for i, (start, end, label) in df.iterrows()]
    ann = convert_structure_annotation(ann, Fs=Fs, remove_digits=remove_digits, index=index)
    color_ann = {}
    if len(fn_ann_color) > 0:
        color_ann = get_color_for_annotation_file(fn_ann_color)
        if remove_digits:
            color_ann_reduced = {}
            for key, value in color_ann.items():
                key_new = ''.join([i for i in key if not i.isdigit()])
                color_ann_reduced[key_new] = value
            color_ann = color_ann_reduced
    return ann, color_ann

# Annotation file
filename = 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.csv'
fn_ann = os.path.join('..', 'data', 'C4', filename)

# Read annotations
ann, color_ann = read_structure_annotation(fn_ann, fn_ann_color=filename)
print('Original annotations with time specified in seconds')
print('Annotations:', ann)
print('Colors:', color_ann)
fig, ax = libfmp.b.plot_segments(ann, figsize=(8, 1.2), colors=color_ann, time_label='Time (seconds)')
plt.show()

# Read and convert annotations
Fs = 2
ann, color_ann = read_structure_annotation(fn_ann, fn_ann_color=filename, Fs=Fs, remove_digits=True, index=True)
print('Converted annotations (Fs = %d) with reduced labels (removing digits)'%Fs)
print('Annotations:', ann)
print('Colors:', color_ann)
fig, ax = libfmp.b.plot_segments(ann, figsize=(8, 1.2), colors=color_ann, time_label='Time (frames)')
plt.show()

# %% [markdown]
# ## Musical Dimensions
#
# The applicability of the different segmentation principles very much depends on the
# musical and acoustic properties of the audio signal to be analyzed. The first step
# in automated structure analysis is to transform the given music recording into a
# suitable **feature representation** that captures the relevant musical properties.

# %%
# Annotations
filename = 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.csv'
fn_ann = os.path.join('..', 'data', 'C4', filename)
ann, color_ann = read_structure_annotation(fn_ann, fn_ann_color=filename, Fs=1, remove_digits=False)

# Waveform
fn_wav = os.path.join('..', 'data', 'C4', 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)
x_dur = x.shape[0]/Fs

# Visualization
fig, ax = plt.subplots(2, 1, gridspec_kw={'width_ratios': [1],
                                          'height_ratios': [1, 0.5]}, figsize=(8, 2.5))
libfmp.b.plot_signal(x, Fs, ax=ax[0], title='Waveform of audio signal')

libfmp.b.plot_segments_overlay(ann, ax=ax[0], time_max=x_dur,
                print_labels=False,  label_ticks=False, edgecolor='gray',
                colors = color_ann, fontsize=10, alpha=0.1)

libfmp.b.plot_segments(ann, ax=ax[1], time_max=x_dur,
                       colors=color_ann, time_label='Time (seconds)')
plt.tight_layout()

# %% [markdown]
# ## Chromagram Representation
#
# First, the **chroma-based representation** relates to **harmonic** and **melodic
# properties** of the music recording. The patterns visible in the chromagram reveal
# important structural information.

# %%
# Chromagram
N, H = 4096, 2048
chromagram = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=2, hop_length=H, n_fft=N)

filt_len = 41
down_sampling = 10
filt_kernel = np.ones([1,filt_len])
chromagram_smooth =  signal.convolve(chromagram, filt_kernel, mode='same')/filt_len
chromagram_smooth = chromagram_smooth[:,::down_sampling]
chromagram_smooth, Fs_smooth = \
    libfmp.c3.smooth_downsample_feature_sequence(chromagram,
                        Fs/H, filt_len=filt_len, down_sampling=down_sampling)

# Visualization
fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.03],
                                          'height_ratios': [2, 2, 0.5]}, figsize=(9, 5))
libfmp.b.plot_chromagram(chromagram, Fs=Fs/H, ax=[ax[0,0], ax[0,1]],
                         chroma_yticks = [0,4,7,11],
                         title='Chromagram (resolution %0.1f Hz)'%(Fs/H),
                         ylabel='Chroma', colorbar=True);
libfmp.b.plot_chromagram(chromagram_smooth, Fs_smooth, ax=[ax[1,0], ax[1,1]],
                         chroma_yticks = [0,4,7,11],
                         title='Smoothed chromagram (resolution %0.1f Hz)'%Fs_smooth,
                         ylabel='Chroma', colorbar=True);
libfmp.b.plot_segments(ann, ax=ax[2,0], time_max=x_dur,
                       colors=color_ann, time_label='Time (seconds)')
ax[2,1].axis('off')
plt.tight_layout()

# %% [markdown]
# ## MFCC Representation
#
# Besides melody and harmony, the instrumentation and timbral characteristics are of
# great importance for the human perception of music structure. MFCC-based features
# are a mid-level representation that somehow correlate to aspects such as
# **instrumentation** and **timbre**.

# %%
# MFCC
N, H = 4096, 2048
X_MFCC = librosa.feature.mfcc(y=x, sr=Fs, hop_length=H, n_fft=N)
coef = np.arange(4,15)
X_MFCC_upper = X_MFCC[coef,:]

# Visualization
fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.03],
                                          'height_ratios': [2, 2, 0.5]}, figsize=(9, 5))
libfmp.b.plot_matrix(X_MFCC, Fs=Fs/H, ax=[ax[0,0], ax[0,1]],
                     title='MFCC (coefficents 0 to 19)', ylabel='', colorbar=True);
ax[0,0].set_yticks([0, 10, 19])
libfmp.b.plot_matrix(X_MFCC_upper, Fs=Fs/H, ax=[ax[1,0], ax[1,1]],
                     title='MFFC (coefficents 4 to 14)', ylabel='', colorbar=True);
ax[1,0].set_yticks([0, 5, 10])
ax[1,0].set_yticklabels(coef[0] + [0, 5, 10])
libfmp.b.plot_segments(ann, ax=ax[2,0], time_max=x_dur,
                       colors=color_ann, time_label='Time (seconds)');
ax[2,1].axis('off')
plt.tight_layout()

# %% [markdown]
# ## Tempogram Representation
#
# In music structure analysis, tempo and beat information may also be used in combination
# with homogeneity-based segmentation approaches. A **tempogram** is a mid-level
# representation that encodes local **tempo information**.

# %%
# Tempogram
nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512, gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs_nov, N=1000, H=100, Theta=np.arange(30, 601))

octave_bin = 30
tempogram_F = np.abs(X)
output = libfmp.c6.compute_cyclic_tempogram(tempogram_F, F_coef_BPM, octave_bin=octave_bin)
tempogram_cyclic_F = output[0]
F_coef_scale = output[1]
tempogram_cyclic_F = libfmp.c3.normalize_feature_sequence(tempogram_cyclic_F, norm='max')

output = libfmp.c6.compute_tempogram_autocorr(nov, Fs_nov, N=500, H=100, norm_sum=False,
                                              Theta=np.arange(30, 601))
tempogram_A = output[0]
T_coef = output[1]
F_coef_BPM = output[2]

output = libfmp.c6.compute_cyclic_tempogram(tempogram_A, F_coef_BPM, octave_bin=octave_bin)
tempogram_cyclic_A = output[0]
F_coef_scale = output[1]
tempogram_cyclic_A = libfmp.c3.normalize_feature_sequence(tempogram_cyclic_A, norm='max')

# Visualization
fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.03],
                                          'height_ratios': [2, 2, 0.5]}, figsize=(9, 5))

libfmp.b.plot_matrix(tempogram_cyclic_F, T_coef=T_coef, ax=[ax[0,0], ax[0,1]],
                     title='Fourier-based cyclic tempogram', ylabel='Scaling',
                     colorbar=True, clim=[0, 1])
libfmp.c6.set_yticks_tempogram_cyclic(ax[0,0], octave_bin, F_coef_scale, num_tick=5)

libfmp.b.plot_matrix(tempogram_cyclic_A, T_coef=T_coef, ax=[ax[1,0], ax[1,1]],
                     title='Autocorrelation-based cyclic tempogram', ylabel='Scaling',
                     colorbar=True, clim=[0, 1])
libfmp.c6.set_yticks_tempogram_cyclic(ax[1,0], octave_bin, F_coef_scale, num_tick=5)

libfmp.b.plot_segments(ann, ax=ax[2,0], time_max=x_dur,
                       colors=color_ann, time_label='Time (seconds)')
ax[2,1].axis('off')

plt.tight_layout()

# %% [markdown]
# ## Further Notes
#
# Besides the various musical dimensions, there is another aspect one should keep in
# mind when looking for suitable feature representations: the **temporal dimension**.
# In all of the above-mentioned feature representations, an analysis window is shifted
# over the music signal. Obviously, the length of the analysis window as well as the
# hop size parameter have a crucial influence on the quality of the feature representation.
#
# In summary, a suitable choice of feature representations and parameter settings very
# much depends on the application context. Humans constantly and often unconsciously
# adapt themselves to the musical and acoustic characteristics of what they listen to.
# The richness and variety of musical structures make computational structure analysis
# a challenging problem.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

# %% [markdown]
# # Cyclic Tempogram
#
# Following Section 6.2.4 of [Müller, FMP, Springer 2015], we introduce in this notebook the
# concept of cyclic tempograms.

# %% [markdown]
# ## Definition (Continuous Case)
#
# The various pulse levels can be seen in analogy to the existence of harmonics in the pitch
# context. To reduce the effects of harmonics, we introduced the concept of chroma-based audio
# features. By identifying pitches that differ by one or several octaves, we obtained a cyclic
# mid-level representation that captures harmonic information while being robust to changes in
# timbre. Inspired by the concept of chroma features, we now introduce the concept of cyclic
# tempograms. The idea is to form tempo equivalence classes by identifying tempi that differ
# by a power of two. More precisely, we say that two tempi $\tau_1$ and $\tau_2$ are **octave
# equivalent**, if they are related by $\tau_1 = 2^{k} \tau_2$ for some $k\in \mathbb{Z}$.

# %% [markdown]
# ## Definition (Discrete Case)
#
# In practice, one can compute a cyclic tempogram $\mathcal{C}_{\tau_0}$ only for a finite
# number of parameters $s\in[1,2)$. To compute a value $\mathcal{C}_{\tau_0}(n,s)$ one needs
# to sum the values $\mathcal{T}(n,\tau)$ for tempo parameters
# $\tau\in \{s\cdot\tau_0\cdot2^k\,\mid\,k\in\mathbb{Z}\}$. One requires a **log-tempo axis**
# for computing a cyclic tempogram.

# %% [markdown]
# ## Cyclic Fourier Tempogram

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
import libfmp.c6
import libfmp.c4

%matplotlib inline

def compute_cyclic_tempogram(tempogram, F_coef_BPM, tempo_ref=30,
                             octave_bin=40, octave_num=4):
    """Compute cyclic tempogram

    Notebook: C6/C6S2_TempogramCyclic.ipynb

    Args:
        tempogram (np.ndarray): Input tempogram
        F_coef_BPM (np.ndarray): Tempo axis (BPM)
        tempo_ref (float): Reference tempo (BPM) (Default value = 30)
        octave_bin (int): Number of bins per tempo octave (Default value = 40)
        octave_num (int): Number of tempo octaves to be considered (Default value = 4)

    Returns:
        tempogram_cyclic (np.ndarray): Cyclic tempogram tempogram_cyclic
        F_coef_scale (np.ndarray): Tempo axis with regard to scaling parameter
        tempogram_log (np.ndarray): Tempogram with logarithmic tempo axis
        F_coef_BPM_log (np.ndarray): Logarithmic tempo axis (BPM)
    """
    F_coef_BPM_log = tempo_ref * np.power(2, np.arange(0, octave_num*octave_bin)/octave_bin)
    F_coef_scale = np.power(2, np.arange(0, octave_bin)/octave_bin)
    tempogram_log = interp1d(F_coef_BPM, tempogram, kind='linear', axis=0, fill_value='extrapolate')(F_coef_BPM_log)
    K = len(F_coef_BPM_log)
    tempogram_cyclic = np.zeros((octave_bin, tempogram.shape[1]))
    for m in np.arange(octave_bin):
        tempogram_cyclic[m, :] = np.mean(tempogram_log[m:K:octave_bin, :], axis=0)
    return tempogram_cyclic, F_coef_scale, tempogram_log, F_coef_BPM_log

def set_yticks_tempogram_cyclic(ax, octave_bin, F_coef_scale, num_tick=5):
    """Set yticks with regard to scaling parameter

    Notebook: C6/C6S2_TempogramCyclic.ipynb

    Args:
        ax (mpl.axes.Axes): Figure axis
        octave_bin (int): Number of bins per tempo octave
        F_coef_scale (np.ndarra): Tempo axis with regard to scaling parameter
        num_tick (int): Number of yticks (Default value = 5)
    """
    yticks = np.arange(0, octave_bin, octave_bin // num_tick)
    ax.set_yticks(yticks)
    ax.set_yticklabels(F_coef_scale[yticks].astype((np.unicode_, 4)))

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_ClickTrack-BPM110-130.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512,
                                                 gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs_nov,
                                                            N=500, H=10,
                                                            Theta=np.arange(30, 601))
tempogram = np.abs(X)
tempo_ref = 30
octave_bin = 40
octave_num = 4
output = compute_cyclic_tempogram(tempogram, F_coef_BPM,
              tempo_ref=tempo_ref, octave_bin=octave_bin, octave_num=octave_num)
tempogram_cyclic = output[0]
F_coef_scale = output[1]
tempogram_log = output[2]
F_coef_BPM_log = output[3]

fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1.5, 1.5, 1]}, figsize=(7, 8))

# Fourier tempogram
im_fig, im_ax, im = libfmp.b.plot_matrix(tempogram, ax=[ax[0]],T_coef=T_coef, F_coef=F_coef_BPM,
                                         title='Fourier tempogram',
                                         ylabel='Tempo (BPM)', colorbar=True);
ax[0].set_yticks([F_coef_BPM[0],100, 200, 300, 400, 500, F_coef_BPM[-1]]);

# Fourier tempogram with log tempo axis
im_fig, im_ax, im = libfmp.b.plot_matrix(tempogram_log, ax=[ax[1]], T_coef=T_coef,
                                         title='Fourier tempogram with log-tempo axis',
                                         ylabel='Tempo (BPM)', colorbar=True);
yticks = np.arange(octave_num) * octave_bin
ax[1].set_yticks(yticks)
ax[1].set_yticklabels(F_coef_BPM_log[yticks].astype(int));

# Cyclic Fourier tempogram
im_fig, im_ax, im = libfmp.b.plot_matrix(tempogram_cyclic, ax=[ax[2]], T_coef=T_coef,
                                         title='Cyclic Fourier tempogram',
                                         ylabel='Scaling', colorbar=True);
set_yticks_tempogram_cyclic(ax[2], octave_bin, F_coef_scale, num_tick=5)
plt.tight_layout()

# %% [markdown]
# ## Cyclic Autocorrelation Tempogram

# %%
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_ClickTrack-BPM110-130.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512,
                                                 gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

N = 500
H = 10
Theta = np.arange(30, 601)
output = libfmp.c6.compute_tempogram_autocorr(nov, Fs_nov, N=N, H=H,
                                              norm_sum=False, Theta=np.arange(30, 601))
tempogram = output[0]
T_coef = output[1]
F_coef_BPM = output[2]

tempo_ref = 30
octave_bin = 40
octave_num = 4
output = compute_cyclic_tempogram(tempogram, F_coef_BPM, tempo_ref=tempo_ref,
                                  octave_bin=octave_bin, octave_num=octave_num)
tempogram_cyclic = output[0]
F_coef_scale = output[1]
tempogram_log = output[2]
F_coef_BPM_log = output[3]

fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1.5, 1.5, 1]}, figsize=(7, 8))

# Autocorrelation tempogram
im_fig, im_ax, im = libfmp.b.plot_matrix(tempogram, ax=[ax[0]], T_coef=T_coef,
                                         F_coef=F_coef_BPM,
                                         figsize=(6,3), ylabel='Tempo (BPM)', colorbar=True,
                                         title='Autocorrelation tempogram');
ax[0].set_yticks([Theta[0],100, 200, 300, 400, 500, Theta[-1]]);

# Autocorrelation tempogram with log tempo axis
im_fig, im_ax, im = libfmp.b.plot_matrix(tempogram_log, ax=[ax[1]], T_coef=T_coef,
                                         figsize=(6,3), ylabel='Tempo (BPM)', colorbar=True,
                                         title='Autocorrelation tempogram with log-tempo axis');
yticks = np.arange(octave_num) * octave_bin
ax[1].set_yticks(yticks)
ax[1].set_yticklabels(F_coef_BPM_log[yticks].astype(int));

# Cyclic autocorrelation tempogram
im_fig, im_ax, im = libfmp.b.plot_matrix(tempogram_cyclic, ax=[ax[2]], T_coef=T_coef,
                                         figsize=(6,2), ylabel='Scaling', colorbar=True,
                                         title='Cyclic autocorrelation tempogram', );
set_yticks_tempogram_cyclic(ax[2], octave_bin, F_coef_scale, num_tick=5)
plt.tight_layout()

# %% [markdown]
# ## Tempo Harmonics and Subharmonics
#
# The Fourier tempogram emphasizes tempo harmonics, while the autocorrelation tempogram
# emphasizes tempo subharmonics. These properties are also reflected by the cyclic versions
# of the tempograms.

# %%
def plot_tempogram_Fourier_autocor(tempogram_F, tempogram_A, T_coef, F_coef_BPM,
                                   octave_bin, title_F, title_A, norm=None):
    """Visualize Fourier-based and autocorrelation-based tempogram
    Notebook: C6/C6S2_TempogramCyclic.ipynb"""
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1,1]}, figsize=(12, 1.5))

    output = compute_cyclic_tempogram(tempogram_F, F_coef_BPM, octave_bin=octave_bin)
    tempogram_cyclic_F = output[0]
    F_coef_scale = output[1]
    if norm is not None:
        tempogram_cyclic_F = libfmp.c3.normalize_feature_sequence(tempogram_cyclic_F,
                                                                  norm=norm)
    libfmp.b.plot_matrix(tempogram_cyclic_F, T_coef=T_coef, ax=[ax[0]],
                         title=title_F, ylabel='Scaling', colorbar=True);
    set_yticks_tempogram_cyclic(ax[0], octave_bin, F_coef_scale, num_tick=5)

    output = compute_cyclic_tempogram(tempogram_A, F_coef_BPM, octave_bin=octave_bin)
    tempogram_cyclic_A  = output[0]
    F_coef_scale = output[1]
    if norm is not None:
        tempogram_cyclic_A = libfmp.c3.normalize_feature_sequence(tempogram_cyclic_A,
                                                                  norm=norm)
    libfmp.b.plot_matrix(tempogram_cyclic_A, T_coef=T_coef, ax=[ax[1]],
                         title=title_A, ylabel='Scaling', colorbar=True);
    set_yticks_tempogram_cyclic(ax[1], octave_bin, F_coef_scale, num_tick=5)

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_ClickTrack-BPM110-130.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)
nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512,
                                                 gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

N = 500
H = 10
Theta = np.arange(30, 601)
X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs_nov, N=N, H=H,
                                                            Theta=Theta)
tempogram_F = np.abs(X)
output = libfmp.c6.compute_tempogram_autocorr(nov, Fs_nov, N=N, H=H,
                                              Theta=Theta, norm_sum=False)
tempogram_A = output[0]

octave_bin=40
title_F = r'Fourier ($M=%d$)'%octave_bin
title_A = r'Autocorrelation ($M=%d$)'%octave_bin
plot_tempogram_Fourier_autocor(tempogram_F, tempogram_A, T_coef, F_coef_BPM,
                               octave_bin, title_F, title_A)

octave_bin=40
title_F = r'Fourier ($M=%d$, max-normalized)'%octave_bin
title_A = r'Autocorrelation ($M=%d$, max-normalized)'%octave_bin
plot_tempogram_Fourier_autocor(tempogram_F, tempogram_A, T_coef, F_coef_BPM,
                               octave_bin, title_F, title_A, norm='max')

octave_bin=15
title_F = r'Fourier ($M=%d$, max-normalized)'%octave_bin
title_A = r'Autocorrelation ($M=%d$, max-normalized)'%octave_bin
plot_tempogram_Fourier_autocor(tempogram_F, tempogram_A, T_coef, F_coef_BPM,
                               octave_bin, title_F, title_A, norm='max')

# %% [markdown]
# ## Tempo Features
#
# The cyclic tempogram representations are the tempo-based counterparts of harmony-based
# chromagram representations. Compared with standard tempograms, the cyclic versions are more
# robust to ambiguities that are caused by the various pulse levels.

# %% [markdown]
# ## Example: Brahms

# %%
# Annotation
filename = 'FMP_C6_Audio_Brahms_HungarianDances-05_Ormandy.csv'
fn_ann = os.path.join('..', 'data', 'C6', filename)
ann, color_ann = libfmp.c4.read_structure_annotation(fn_ann, fn_ann_color=filename,
                                                     Fs=1, remove_digits=False)

# Audio file
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_Brahms_HungarianDances-05_Ormandy.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512,
                                                 gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

octave_bin = 15
X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs_nov, N=500, H=50,
                                                            Theta=np.arange(30, 601))
tempogram_F = np.abs(X)

tempogram_A, T_coef, F_coef_BPM, _, _ = libfmp.c6.compute_tempogram_autocorr(nov, Fs_nov,
                                                                             N=500, H=50,
                                                                             norm_sum=False,
                                                                             Theta=np.arange(30, 601))

fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.03],
                                          'height_ratios': [2, 2, 1]}, figsize=(8, 5))

output = compute_cyclic_tempogram(tempogram_F, F_coef_BPM, octave_bin=octave_bin)
tempogram_cyclic_F = output[0]
F_coef_scale = output[1]

tempogram_cyclic_F = libfmp.c3.normalize_feature_sequence(tempogram_cyclic_F, norm='max')
libfmp.b.plot_matrix(tempogram_cyclic_F, T_coef=T_coef, ax=[ax[0,0], ax[0,1]], clim=[0,1],
                     title='Fourier ($M=15$, max-normalized)',
                     ylabel='Scaling', colorbar=True);
set_yticks_tempogram_cyclic(ax[0,0], octave_bin, F_coef_scale, num_tick=5)

output = compute_cyclic_tempogram(tempogram_A, F_coef_BPM, octave_bin=octave_bin)
tempogram_cyclic_A = output[0]
F_coef_scale = output[1]

tempogram_cyclic_A = libfmp.c3.normalize_feature_sequence(tempogram_cyclic_A, norm='max')
libfmp.b.plot_matrix(tempogram_cyclic_A, T_coef=T_coef, ax=[ax[1,0], ax[1,1]], clim=[0,1],
                     title='Autocorrelation ($M=15$, max-normalized)',
                     ylabel='Scaling', colorbar=True);
set_yticks_tempogram_cyclic(ax[1,0], octave_bin, F_coef_scale, num_tick=5)

libfmp.b.plot_segments(ann, ax=ax[2,0], time_max=(x.shape[0])/Fs,
                       colors=color_ann, time_label='Time (seconds)')
ax[2,1].axis('off')

plt.tight_layout()

# %% [markdown]
# ## Example: Zager and Evans

# %%
# Annotation
filename = 'FMP_C6_Audio_ZagerEvans_InTheYear2525.csv'
fn_ann = os.path.join('..', 'data', 'C6', filename)
ann, color_ann = libfmp.c4.read_structure_annotation(fn_ann, fn_ann_color=filename,
                                                     Fs=1, remove_digits=False)

# Audio file
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_Audio_ZagerEvans_InTheYear2525.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512,
                                                 gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

octave_bin = 15
X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs_nov, N=500, H=50,
                                                            Theta=np.arange(30, 601))
tempogram_F = np.abs(X)

tempogram_A, T_coef, F_coef_BPM, _, _ = libfmp.c6.compute_tempogram_autocorr(nov, Fs_nov,
                                                                             N=500, H=50,
                                                                             norm_sum=False,
                                                                             Theta=np.arange(30, 601))

fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.03],
                                          'height_ratios': [2, 2, 1]}, figsize=(8, 5))

output = compute_cyclic_tempogram(tempogram_F, F_coef_BPM, octave_bin=octave_bin)
tempogram_cyclic_F = output[0]
F_coef_scale = output[1]

tempogram_cyclic_F = libfmp.c3.normalize_feature_sequence(tempogram_cyclic_F, norm='max')
libfmp.b.plot_matrix(tempogram_cyclic_F, T_coef=T_coef, ax=[ax[0,0], ax[0,1]], clim=[0,1],
                     title='Fourier ($M=%d$, max-normalized)'%octave_bin,
                     ylabel='Scaling', colorbar=True);
set_yticks_tempogram_cyclic(ax[0,0], octave_bin, F_coef_scale, num_tick=5)

output = compute_cyclic_tempogram(tempogram_A, F_coef_BPM, octave_bin=octave_bin)
tempogram_cyclic_A = output[0]
F_coef_scale = output[1]

tempogram_cyclic_A = libfmp.c3.normalize_feature_sequence(tempogram_cyclic_A, norm='max')
libfmp.b.plot_matrix(tempogram_cyclic_A, T_coef=T_coef, ax=[ax[1,0], ax[1,1]], clim=[0,1],
                     title='Autocorrelation ($M=%d$, max-normalized)'%octave_bin,
                     ylabel='Scaling', colorbar=True);
set_yticks_tempogram_cyclic(ax[1,0], octave_bin, F_coef_scale, num_tick=5)

libfmp.b.plot_segments(ann, ax=ax[2,0], time_max=(x.shape[0])/Fs,
                       colors=color_ann, time_label='Time (seconds)')
ax[2,1].axis('off')

plt.tight_layout()

# %% [markdown]
# ## Further Notes
#
# The idea of tempo-based feature representations is to capture local periodicities occurring
# in the underlying signal. There are many ways for computing such time–tempo representations
# known as **tempograms**, **rhythmograms**, or **beat spectrograms**. In this notebook we
# considered **cyclic versions** (similar to chroma-based features), which possess a high degree
# of robustness to pulse level switches. Rather than measuring the specific tempo of a local
# section of a given recording, **cyclic tempogram features** allow for capturing the existence
# or absence of a notion of tempo—a kind of **tempo salience**.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

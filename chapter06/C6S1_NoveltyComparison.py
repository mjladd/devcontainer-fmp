# %% [markdown]
# # Novelty: Comparison of Approaches
#
# In this notebook, we compare different approaches for computing novelty functions,
# see Section 6.1 of [Müller, FMP, Springer 2015].

# %% [markdown]
# ## Approaches for Novelty Computation
#
# We compare four different approaches for computing a novelty function:
# * Energy-based novelty approach
# * Spectral-based novelty approach
# * Phase-based novelty approach
# * Complex-domain novelty approach

# %%
import numpy as np
import os, sys, librosa
from scipy import signal
from scipy.interpolate import interp1d
from scipy import ndimage
from matplotlib import pyplot as plt
import IPython.display as ipd

sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c6

%matplotlib inline


fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_F01_Queen.csv')
ann, label_keys = libfmp.c6.read_annotation_pos(fn_ann)

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F01_Queen.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)
x_dur = len(x)/Fs

nov_dic = {}

nov, Fs_nov = libfmp.c6.compute_novelty_energy(x, Fs=Fs, gamma=None)
nov_dic.update( {0 : [nov, Fs_nov, r'Energy-based novelty function (Fs = %d)'%Fs_nov]} )

nov, Fs_nov = libfmp.c6.compute_novelty_energy(x, Fs=Fs, gamma=1000)
nov_dic.update( {1 : [nov, Fs_nov, 'Energy-based novelty function with compression (Fs = %d)'%Fs_nov]} )

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs)
nov_dic.update( {2 : [nov, Fs_nov, 'Spectral-based novelty function (Fs = %d)'%Fs_nov]} )

nov, Fs_nov = libfmp.c6.compute_novelty_phase(x, Fs=Fs)
nov_dic.update( {3 : [nov, Fs_nov, 'Phase-based novelty function (Fs = %d)'%Fs_nov]} )

nov, Fs_nov = libfmp.c6.compute_novelty_complex(x, Fs=Fs)
nov_dic.update( {4 : [nov, Fs_nov, 'Complex-domain novelty function (Fs = %d)'%Fs_nov]} )

for k in nov_dic:
    fig, ax, line = libfmp.b.plot_signal(nov_dic[k][0], Fs=nov_dic[k][1],
                        color='k', title=nov_dic[k][2])
    libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                nontime_axis=True, time_min=0, time_max=x_dur)

# %% [markdown]
# ## Time Axis Resampling
#
# The different approaches may lead to novelty functions with different feature rates.
# To directly compare or combine these novelty functions, we introduce a resampling
# approach using linear interpolation.

# %%
def resample_signal(x_in, Fs_in, Fs_out=100, norm=True, time_max_sec=None, sigma=None):
    """Resample and smooth signal

    Notebook: C6/C6S1_NoveltyComparison.ipynb

    Args:
        x_in (np.ndarray): Input signal
        Fs_in (scalar): Sampling rate of input signal
        Fs_out (scalar): Sampling rate of output signal (Default value = 100)
        norm (bool): Apply max norm (if norm==True) (Default value = True)
        time_max_sec (float): Duration of output signal (given in seconds) (Default value = None)
        sigma (float): Standard deviation for smoothing Gaussian kernel (Default value = None)

    Returns:
        x_out (np.ndarray): Output signal
        Fs_out (scalar): Feature rate of output signal
    """
    if sigma is not None:
        x_in = ndimage.gaussian_filter(x_in, sigma=sigma)
    T_coef_in = np.arange(x_in.shape[0]) / Fs_in
    time_in_max_sec = T_coef_in[-1]
    if time_max_sec is None:
        time_max_sec = time_in_max_sec
    N_out = int(np.ceil(time_max_sec*Fs_out))
    T_coef_out = np.arange(N_out) / Fs_out
    if T_coef_out[-1] > time_in_max_sec:
        x_in = np.append(x_in, [0])
        T_coef_in = np.append(T_coef_in, [T_coef_out[-1]])
    x_out = interp1d(T_coef_in, x_in, kind='linear')(T_coef_out)
    if norm:
        x_max = max(x_out)
        if x_max > 0:
            x_out = x_out / max(x_out)
    return x_out, Fs_out

nov, Fs_nov = libfmp.c6.compute_novelty_complex(x, Fs)
libfmp.b.plot_signal(nov, Fs=1, xlabel='Time (samples)', color='k',
        title='Fs = %0.1f'%Fs_nov);

nov_out, Fs_out = resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)
libfmp.b.plot_signal(nov_out, Fs=1, xlabel='Time (samples)', color='k',
        title='Fs = %0.1f'%Fs_out);

# %% [markdown]
# ## Effect of Smoothing on Resampling
#
# Reducing the feature rate via linear interpolation may be problematic for novelty
# functions with peak-like structures. Applying a smoothing filter prior to resampling
# can help preserve the peak structure.

# %%
fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1,1], 'height_ratios': [1,1]},
                       figsize=(10,4))

nov, Fs_nov = libfmp.c6.compute_novelty_phase(x, Fs)
libfmp.b.plot_signal(nov, Fs=1, ax=ax[0,0], xlabel='Time (samples)', color='k',
        title='Original novelty function (Fs = %0.1f)'%Fs_nov)

nov_out, Fs_out = resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)
libfmp.b.plot_signal(nov_out, ax=ax[1,0], Fs=1, xlabel='Time (samples)', color='k',
        title='Novelty function after resampling (Fs = %0.1f)'%Fs_out);

Fs_out = Fs_nov
nov_smooth, Fs_out = resample_signal(nov, Fs_in=Fs_nov, Fs_out=Fs_out, sigma=4)
libfmp.b.plot_signal(nov_smooth, Fs=1, ax=ax[0,1], xlabel='Time (samples)', color='k',
        title='Smoothed novelty function (Fs = %0.1f)'%Fs_nov)

Fs_out = 100
nov_smooth_out, Fs_out = resample_signal(nov, Fs_in=Fs_nov, Fs_out=Fs_out, sigma=2)
libfmp.b.plot_signal(nov_smooth_out, ax=ax[1,1], Fs=1, xlabel='Time (samples)', color='k',
        title='Smoothed novelty function after resampling (Fs = %0.1f)'%Fs_out);

plt.tight_layout()

# %% [markdown]
# ## Matrix-Based Visualization and Averaging
#
# After converting different novelty functions to a common discrete time axis, one can
# easily visualize these functions in a color-coded form and compute their average.

# %%
def average_nov_dic(nov_dic, time_max_sec, Fs_out=100, norm=True, sigma=None):
    """Average resampled set of novelty functions

    Notebook: C6/C6S1_NoveltyComparison.ipynb

    Args:
        nov_dic (dict): Dictionary of novelty functions
        time_max_sec (float): Duration of output signals (given in seconds)
        Fs_out (scalar): Sampling rate of output signal (Default value = 100)
        norm (bool): Apply max norm (if norm==True) (Default value = True)
        sigma (float): Standard deviation for smoothing Gaussian kernel (Default value = None)

    Returns:
        nov_matrix (np.ndarray): Matrix containing resampled output signal (last one is average)
        Fs_out (scalar): Sampling rate of output signals
    """
    nov_num = len(nov_dic)
    N_out = int(np.ceil(time_max_sec*Fs_out))
    nov_matrix = np.zeros([nov_num + 1, N_out])
    for k in range(nov_num):
        nov = nov_dic[k][0]
        Fs_nov = nov_dic[k][1]
        nov_out, Fs_out = resample_signal(nov, Fs_in=Fs_nov, Fs_out=Fs_out,
                                          time_max_sec=time_max_sec, sigma=sigma)
        nov_matrix[k, :] = nov_out
    nov_average = np.sum(nov_matrix, axis=0)/nov_num
    if norm:
        max_value = np.max(nov_average)
        if max_value > 0:
            nov_average = nov_average / max_value
    nov_matrix[nov_num, :] = nov_average
    return nov_matrix, Fs_out

cmap = libfmp.b.compressed_gray_cmap(alpha=1)
Fs_out = 100
nov_matrix, Fs_out = average_nov_dic(nov_dic, time_max_sec=x_dur, Fs_out=Fs_out)

plt.figure(figsize=[8,3])
ax = plt.subplot(1,1,1)
im = ax.imshow(nov_matrix, cmap=cmap, aspect='auto', clim=[0,1],
          extent=[0, x_dur, nov_matrix.shape[0]+0.5, 0.5], interpolation='nearest')
ax.set_xlabel('Time (seconds)')
ax.set_yticks([1,2,3,4,5,6])
ax.set_yticklabels([r'Energy', r'EnergyLog', r'Spectral', r'Phase', r'Complex', r'Average'])
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                    nontime_axis=True, time_min=0, time_max=x_dur);
plt.colorbar(im)
plt.tight_layout()

# %% [markdown]
# ## Smoothed Novelty Functions

# %%
Fs_out = 100
sigma = 2
nov_matrix_smooth, Fs_out = average_nov_dic(nov_dic, time_max_sec=x_dur,
                                            Fs_out=Fs_out, sigma=sigma)

plt.figure(figsize=[8,3])
ax = plt.subplot(1,1,1)
im = ax.imshow(nov_matrix_smooth, cmap=cmap, aspect='auto', clim=[0,1],
          extent=[0, x_dur, nov_matrix.shape[0]+0.5, 0.5], interpolation='nearest')
ax.set_xlabel('Time (seconds)')
ax.set_yticks([1,2,3,4,5,6])
ax.set_yticklabels([r'Energy', r'EnergyLog', r'Spectral', r'Phase', r'Complex', r'Average'])
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                    nontime_axis=True, time_min=0, time_max=x_dur);
plt.colorbar(im)
plt.tight_layout()

fig, ax, line = libfmp.b.plot_signal(nov_matrix[-1,:], Fs=Fs_out,
                        color='k', title='Average novelty function without smoothing')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                nontime_axis=True, time_min=0, time_max=x_dur)

fig, ax, line = libfmp.b.plot_signal(nov_matrix_smooth[-1,:], Fs=Fs_out,
                        color='k', title='Average novelty function with smoothing')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                nontime_axis=True, time_min=0, time_max=x_dur);

# %% [markdown]
# ## Example: Note C4
#
# We consider the note C4 played by different instruments (piano, trumpet, violin, flute).

# %%
fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_F04_NoteC4_PTVF.csv')
ann, label_keys = libfmp.c6.read_annotation_pos(fn_ann, label='onset', header=0)

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F04_NoteC4_PTVF.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)
x_dur = len(x)/Fs

nov_dic = {}

nov, Fs_nov = libfmp.c6.compute_novelty_energy(x, Fs=Fs, gamma=None)
nov_dic.update( {0 : [nov, Fs_nov, 'Energy-based novelty function']} )

nov, Fs_nov = libfmp.c6.compute_novelty_energy(x, Fs=Fs, gamma=1000)
nov_dic.update( {1 : [nov, Fs_nov, 'Energy-based novelty function with compression']} )

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs)
nov_dic.update( {2 : [nov, Fs_nov, 'Spectral-based novelty function']} )

nov, Fs_nov = libfmp.c6.compute_novelty_phase(x, Fs=Fs)
nov_dic.update( {3 : [nov, Fs_nov, 'Phase-based novelty function']} )

nov, Fs_nov = libfmp.c6.compute_novelty_complex(x, Fs=Fs)
nov_dic.update( {4 : [nov, Fs_nov, 'Complex-domain novelty function']} )

Fs_out = 100
sigma = 8
nov_matrix_smooth, Fs_out = average_nov_dic(nov_dic, time_max_sec=x_dur,
                                            Fs_out=Fs_out, sigma=sigma)

plt.figure(figsize=[8,3])
ax = plt.subplot(1,1,1)
im = ax.imshow(nov_matrix_smooth, cmap=cmap, aspect='auto', clim=[0,1],
          extent=[0, x_dur, nov_matrix.shape[0]+0.5, 0.5], interpolation='nearest')
ax.set_xlabel('Time (seconds)')
ax.set_yticks([1,2,3,4,5,6])
ax.set_yticklabels([r'Energy', r'EnergyLog', r'Spectral', r'Phase', r'Complex', r'Average'])
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                    nontime_axis=True, time_min=0, time_max=x_dur);
plt.colorbar(im)
plt.tight_layout()

# %% [markdown]
# ## Example: Shostakovich
#
# The spectral-based approach for novelty computation turns out to yield more stable
# and qualitatively better results than the other approaches.

# %%
fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_F07_Shostakovich_Waltz-02-Section_IncreasingTempo.csv')
ann, label_keys = libfmp.c6.read_annotation_pos(fn_ann, label='onset', header=0)

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F07_Shostakovich_Waltz-02-Section_IncreasingTempo.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)
x_dur = len(x)/Fs

N, H = 2048, 512
gamma = 10

figsize=(8,2)

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=N, H=H, gamma=gamma)
fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, figsize=figsize, color='k',
    title='Spectral-based novelty function');
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                nontime_axis=True, time_min=0, time_max=x_dur);

nov, Fs_nov = libfmp.c6.compute_novelty_energy(x, Fs=Fs, gamma=None)
fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, figsize=figsize, color='k',
    title='Energy-based novelty function');
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                nontime_axis=True, time_min=0, time_max=x_dur);

nov, Fs_nov = libfmp.c6.compute_novelty_phase(x, Fs=Fs)
fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, figsize=figsize, color='k',
    title='Phase-based novelty function');
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                nontime_axis=True, time_min=0, time_max=x_dur);

nov, Fs_nov = libfmp.c6.compute_novelty_complex(x, Fs=Fs)
fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, figsize=figsize, color='k',
    title='Complex-domain novelty function');
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                nontime_axis=True, time_min=0, time_max=x_dur);

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Angel Villar-Corrales.

# %% [markdown]
# # Peak Picking
#
# In this notebook, we compare different peak picking strategies using note onset
# detection as example task.

# %% [markdown]
# ## Peak Picking
#
# The positions of peaks (well-defined local maxima) of the novelty function are good
# indicators for onset positions. Several strategies can help with peak picking:
#
# * Simple smoothing operations to reduce noise-like fluctuations
# * Adaptive thresholding where a peak is selected only when exceeding a local average
# * Imposing a constraint on the minimal distance between two subsequent peak positions

# %% [markdown]
# ## Running Example: Shostakovich
#
# We use a spectral-based novelty function computed for the Shostakovich example.

# %%
import numpy as np
import os, sys, librosa
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import filters
from matplotlib import pyplot as plt
import IPython.display as ipd

sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c6
%matplotlib inline

fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_F07_Shostakovich_Waltz-02-Section_IncreasingTempo.csv')
ann, label_keys = libfmp.c6.read_annotation_pos(fn_ann, label='onset', header=0)

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F07_Shostakovich_Waltz-02-Section_IncreasingTempo.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)
x_duration = len(x)/Fs

N, H = 2048, 512
gamma = 10
nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=N, H=H, gamma=gamma)
figsize=(8,2)
fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, figsize=figsize, color='k',
    title='Spectral-based novelty function with annotated beat positions (low difficulty)');
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                nontime_axis=True, time_min=0, time_max=x_duration);

N, H = 1024, 256
gamma = 10
nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=N, H=H, gamma=gamma)
figsize=(8,2)
fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, figsize=figsize, color='k',
    title='Spectral-based novelty function with annotated beat positions (medium difficulty)');
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                nontime_axis=True, time_min=0, time_max=x_duration);

# %% [markdown]
# ## Peak Picking: Simple Approach
#
# A simple peak picking approach that looks for all turning points where the curve
# changes from increasing to decreasing.

# %%
def peak_picking_simple(x, threshold=None):
    """Peak picking strategy looking for positions with increase followed by decrease

    Notebook: C6/C6S1_PeakPicking.ipynb

    Args:
        x (np.ndarray): Input function
        threshold (float): Lower threshold for peak to survive

    Returns:
        peaks (np.ndarray): Array containing peak positions
    """
    peaks = []
    if threshold is None:
        threshold = np.min(x) - 1
    for i in range(1, x.shape[0] - 1):
        if x[i - 1] < x[i] and x[i] > x[i + 1]:
            if x[i] >= threshold:
                peaks.append(i)
    peaks = np.array(peaks)
    return peaks

def plot_function_peak_positions(nov, Fs_nov, peaks, title='', figsize=(8,2)):
    peaks_sec = peaks/Fs_nov
    fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, figsize=figsize, color='k', title=title);
    plt.vlines(peaks_sec, 0, 1.1, color='r', linestyle=':', linewidth=1);

peaks = peak_picking_simple(nov, threshold=None)
title='Simple peak picking (without threshold condition)'
plot_function_peak_positions(nov, Fs_nov, peaks, title)

threshold = 0.2
peaks = peak_picking_simple(nov, threshold=threshold)
title='Simple peak picking (threshold=%.1f)'%threshold
plot_function_peak_positions(nov, Fs_nov, peaks, title)
plt.hlines(threshold, 0, x_duration, color='cyan', linewidth=2);
plt.legend(['Novelty function',  'Peak positions',  'Global threshold'],
           loc='upper right', framealpha=1);

# %% [markdown]
# ## Peak Picking: MSAF
#
# A combination of smoothing and adaptive thresholding as used in the MSAF package.

# %%
def peak_picking_MSAF(x, median_len=16, offset_rel=0.05, sigma=4.0):
    """Peak picking strategy following MSFA using an adaptive threshold

    Notebook: C6/C6S1_PeakPicking.ipynb

    Args:
        x (np.ndarray): Input function
        median_len (int): Length of media filter used for adaptive thresholding (Default value = 16)
        offset_rel (float): Additional offset used for adaptive thresholding (Default value = 0.05)
        sigma (float): Variance for Gaussian kernel used for smoothing the novelty function (Default value = 4.0)

    Returns:
        peaks (np.ndarray): Peak positions
        x (np.ndarray): Local threshold
        threshold_local (np.ndarray): Filtered novelty curve
    """
    offset = x.mean() * offset_rel
    x = filters.gaussian_filter1d(x, sigma=sigma)
    threshold_local = filters.median_filter(x, size=median_len) + offset
    peaks = []
    for i in range(1, x.shape[0] - 1):
        if x[i - 1] < x[i] and x[i] > x[i + 1]:
            if x[i] > threshold_local[i]:
                peaks.append(i)
    peaks = np.array(peaks)
    return peaks, x, threshold_local

median_len = 24
sigma = 2
peaks, x_smooth, threshold_local = peak_picking_MSAF(nov, median_len=median_len,
                                                     offset_rel=0.05, sigma=sigma)
title='MSAF peak picking (sigma=%1.0f, median_len=%2.0f)'%(sigma, median_len)
plot_function_peak_positions(nov, Fs_nov, peaks, title)
t = np.arange(nov.shape[0]) / Fs_nov
plt.plot(t, x_smooth, color='magenta', linewidth=2);
plt.plot(t, threshold_local, color='cyan', linewidth=2);
plt.legend(['Novelty function', 'Smoothed Novelty', 'Local threshold'],
           loc='upper right', framealpha=1);

median_len = 16
sigma = 4
peaks, x_smooth, threshold_local = peak_picking_MSAF(nov, median_len=median_len,
                                                     offset_rel=0.05, sigma=sigma)
title='MSAF peak picking (sigma=%1.0f, median_len=%2.0f)'%(sigma, median_len)
plot_function_peak_positions(nov, Fs_nov, peaks, title)
t = np.arange(nov.shape[0]) / Fs_nov
plt.plot(t, x_smooth, color='magenta', linewidth=2);
plt.plot(t, threshold_local, color='cyan', linewidth=2);
plt.legend(['Novelty function',  'Smoothed Novelty', 'Local threshold'],
           loc='upper right', framealpha=1);

# %% [markdown]
# ## Peak Picking: Scipy
#
# The scipy.signal package contains a peak picking function called `find_peaks`.

# %%
prominence = 0.05
peaks = signal.find_peaks(nov, prominence=prominence)[0]
title='Scipy peak picking (prominence=%.2f)'%prominence
plot_function_peak_positions(nov, Fs_nov, peaks, title)

prominence = 0.1
peaks = signal.find_peaks(nov, prominence=prominence)[0]
title='Scipy peak picking (prominence=%.2f)'%prominence
plot_function_peak_positions(nov, Fs_nov, peaks, title)

# %% [markdown]
# ## Height Parameter

# %%
height = (0.1, 0.5)
peaks = signal.find_peaks(nov, height=height)[0]
title='Scipy peak picking (height=(%.2f,%.2f))'%(height[0],height[1])
plot_function_peak_positions(nov, Fs_nov, peaks, title)
plt.hlines(height[0], 0, x_duration, color='cyan', linewidth=2);
plt.hlines(height[1], 0, x_duration, color='cyan', linewidth=2);

height = filters.median_filter(nov, size=8) + 0.1
peaks = signal.find_peaks(nov, height=height)[0]
title='Scipy peak picking (height = local median average)'
plot_function_peak_positions(nov, Fs_nov, peaks, title)
t = np.arange(nov.shape[0]) / Fs_nov
plt.plot(t, height, color='cyan', linewidth=2);
plt.legend(['Novelty function', 'Local threshold'],
           loc='upper right', framealpha=1);

# %% [markdown]
# ## Distance Parameter

# %%
distance = 5
peaks = signal.find_peaks( nov, distance=distance)[0]
title='Scipy peak picking (distance=%2.0f samples, Fs=%3.0f)'%(distance, Fs_nov)
plot_function_peak_positions(nov, Fs_nov, peaks, title)

distance = 15
peaks = signal.find_peaks( nov, distance=distance)[0]
title='Scipy peak picking (distance=%2.0f samples, Fs=%3.0f)'%(distance, Fs_nov)
plot_function_peak_positions(nov, Fs_nov, peaks, title)

# %% [markdown]
# ## Peak Picking: LibROSA
#
# LibROSA provides a peak-picking function `librosa.util.peak_pick` with many
# tunable parameters.

# %%
wait = 5
peaks = librosa.util.peak_pick(nov, pre_max=5, post_max=5, pre_avg=5, post_avg=5,
                               delta=0.01, wait=wait)
title='LibROSA peak picking (wait=%2.0f)'%distance
plot_function_peak_positions(nov, Fs_nov, peaks, title)

wait = 10
peaks = librosa.util.peak_pick(nov, pre_max=5, post_max=5, pre_avg=5, post_avg=5,
                               delta=0.01, wait=wait)
title='LibROSA peak picking (wait=%2.0f)'%wait
plot_function_peak_positions(nov, Fs_nov, peaks, title)

# %% [markdown]
# ## Peak Picking: Boeck
#
# Implementation from Böck, Krebs, and Schedl on Online Capabilities of Onset Detection Methods.

# %%
peaks = libfmp.c6.peak_picking_boeck(nov, threshold=0.01, fps=1,
            include_scores=False, combine=0,
            pre_avg=0.1*Fs_nov, post_avg=0.1*Fs_nov, pre_max=0.1*Fs_nov, post_max=0.1*Fs_nov)
title='Boeck peak picking'
plot_function_peak_positions(nov, Fs_nov, peaks, title)

# %% [markdown]
# ## Peak Picking: Roeder
#
# Peak picking strategy originally implemented by Tido Röder.

# %%
peaks = libfmp.c6.peak_picking_roeder(nov, direction=None, abs_thresh=None,
                                      rel_thresh=None, descent_thresh=None,
                                      tmin=None, tmax=None)
title='Roeder peak picking'
plot_function_peak_positions(nov, Fs_nov, peaks, title)

nov_smooth = filters.gaussian_filter1d(nov, sigma=2)
peaks = libfmp.c6.peak_picking_roeder(nov_smooth, direction=None, abs_thresh=None,
                                      rel_thresh=None, descent_thresh=None,
                                      tmin=None, tmax=None)
title='Roeder peak picking (applied to a smoothed input signal)'
plot_function_peak_positions(nov, Fs_nov, peaks, title)
t = np.arange(nov.shape[0]) / Fs_nov
plt.plot(t, nov_smooth, color='magenta', linewidth=2);
plt.legend(['Novelty function', 'Smoothed novelty'],
           loc='upper right', framealpha=1);

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Angel Villar-Corrales.

# %% [markdown]
# # Transposition and Tuning
#
# In this notebook, we cover aspects of transposition and tuning following Section 3.1.2.2,
# Section 3.1.2.1, Exercise 3.5, and Exercise 3.6 of [Müller, FMP, Springer 2015].

# %% [markdown]
# ## Transposition
#
# In music one often shifts a melody or an entire piece of music to another key—a concept
# that is referred to as **transposition**. Such modifications are often applied to adapt
# the pitch range of a given piece to a different instrument or singer.

# %%
import sys
import os

import numpy as np
from matplotlib import pyplot as plt
import librosa
from scipy.interpolate import interp1d
from scipy import signal

sys.path.append('..')
import libfmp.b
import libfmp.c2
%matplotlib inline

def cyclic_shift(C, shift=1):
    """Cyclically shift a chromagram

    Notebook: C3/C3S1_TranspositionTuning.ipynb

    Args:
        C (np.ndarray): Chromagram
        shift (int): Tranposition shift (Default value = 1)

    Returns:
        C_shift (np.ndarray): Cyclically shifted chromagram
    """
    C_shift = np.roll(C, shift=shift, axis=0)
    return C_shift

x_cmajor, Fs = librosa.load(os.path.join('..', 'data', 'C3', 'FMP_C3_F08_C-major-scale.wav'))
x_emajor, Fs = librosa.load(os.path.join('..', 'data', 'C3', 'FMP_C3_F08_C-major-scale_400-cents-up.wav'))

N = 4096
H = 1024
figsize = (8, 2)
yticks = [0, 4, 7, 11]

C = librosa.feature.chroma_stft(y=x_cmajor, sr=Fs, tuning=0, norm=2, hop_length=H, n_fft=N)
title = 'Chromagram of C-major scale'
libfmp.b.plot_chromagram(C, Fs/H, figsize=figsize, title=title, clim=[0, 1], chroma_yticks=yticks)

C = librosa.feature.chroma_stft(y=x_cmajor, sr=Fs, tuning=0, norm=2, hop_length=H, n_fft=N)
C_shift = cyclic_shift(C, shift=4)
title = 'Chromagram of C-major scale cyclically shifted four semitones upwards'
libfmp.b.plot_chromagram(C_shift, Fs/H, figsize=figsize, title=title, clim=[0, 1], chroma_yticks=yticks)

C = librosa.feature.chroma_stft(y=x_emajor, sr=Fs, tuning=0, norm=2, hop_length=H, n_fft=N)
title = 'Chromagram of E-major scale'
libfmp.b.plot_chromagram(C, Fs/H, figsize=figsize, title=title, clim=[0, 1], chroma_yticks=yticks);

# %% [markdown]
# ## Tuning
#
# While transpositions are pitch shifts on the semitone level, we now discuss global
# frequency deviations on the sub-semitone level. Such deviations may be the result
# of instruments that are tuned lower or higher than the expected reference pitch A4
# with center frequency 440 Hz.

# %%
Fs = 22050
fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3_F08_C-major-scale.wav')
x, Fs = librosa.load(fn_wav)
fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3_F08_C-major-scale_40-cents-up.wav')
x_detune, Fs = librosa.load(fn_wav)

figsize = (8, 2)
yticks = [0, 4, 7, 11]
N = 4096
H = 1024

C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=2, hop_length=H, n_fft=N)
title = 'Chromagram of original signal'
libfmp.b.plot_chromagram(C, Fs/H, figsize=figsize, title=title, clim=[0, 1], chroma_yticks=yticks)

C = librosa.feature.chroma_stft(y=x_detune, sr=Fs, tuning=0, norm=2, hop_length=H, n_fft=N)
title = 'Chromagram of detuned signal'
libfmp.b.plot_chromagram(C, Fs/H, figsize=figsize, title=title, clim=[0, 1], chroma_yticks=yticks);

# %% [markdown]
# To compensate for tuning effects, one needs to perform an additional tuning estimation
# step to adjust the center frequencies of the MIDI pitches and the logarithmic
# partitioning for computing the pitch-based log-frequency spectrogram.

# %%
theta = 40
tuning = theta / 100
C = librosa.feature.chroma_stft(y=x_detune, sr=Fs, tuning=tuning, norm=2, hop_length=H, n_fft=N)
title = 'Chromagram of detuned signal with adjusted chroma bands'
libfmp.b.plot_chromagram(C, Fs/H, figsize=figsize, title=title, clim=[0, 1], chroma_yticks=yticks);

# %% [markdown]
# ## Tuning Estimation
#
# The tuning of musical instruments is usually based on a fixed reference pitch. In
# Western music, one typically uses the **concert pitch** A4 having a frequency of 440 Hz.

# %%
def compute_freq_distribution(x, Fs, N=16384, gamma=100.0, local=True, filt=True, filt_len=101):
    """Compute an overall frequency distribution

    Notebook: C3/C3S1_TranspositionTuning.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate
        N (int): Window size (Default value = 16384)
        gamma (float): Constant for logarithmic compression (Default value = 100.0)
        local (bool): Computes STFT and averages; otherwise computes global DFT (Default value = True)
        filt (bool): Applies local frequency averaging and by rectification (Default value = True)
        filt_len (int): Filter length for local frequency averaging (length given in cents) (Default value = 101)

    Returns:
        v (np.ndarray): Vector representing an overall frequency distribution
        F_coef_cents (np.ndarray): Frequency axis (given in cents)
    """
    if local:
        # Compute an STFT and sum over time
        if N > len(x)//2:
            raise Exception('The signal length (%d) should be twice as long as the window length (%d)' % (len(x), N))
        Y, T_coef, F_coef = libfmp.c2.stft_convention_fmp(x=x, Fs=Fs, N=N, H=N//2, mag=True, gamma=gamma)
        # Error "range() arg 3 must not be zero" occurs when N is too large. Why?
        Y = np.sum(Y, axis=1)
    else:
        # Compute a single DFT for the entire signal
        N = len(x)
        Y = np.abs(np.fft.fft(x)) / Fs
        Y = Y[:N//2+1]
        Y = np.log(1 + gamma * Y)
        # Y = libfmp.c3.log_compression(Y, gamma=100)
        F_coef = np.arange(N // 2 + 1).astype(float) * Fs / N

    # Convert linearly spaced frequency axis in logarithmic axis (given in cents)
    # The minimum frequency F_min corresponds 0 cents.
    f_pitch = lambda p: 440 * 2 ** ((p - 69) / 12)
    p_min = 24               # C1, MIDI pitch 24
    F_min = f_pitch(p_min)   # 32.70 Hz
    p_max = 108              # C8, MIDI pitch 108
    F_max = f_pitch(p_max)   # 4186.01 Hz
    F_coef_log, F_coef_cents = libfmp.c2.compute_f_coef_log(R=1, F_min=F_min, F_max=F_max)
    Y_int = interp1d(F_coef, Y, kind='cubic', fill_value='extrapolate')(F_coef_log)
    v = Y_int / np.max(Y_int)

    if filt:
        # Subtract local average and rectify
        filt_kernel = np.ones(filt_len)
        Y_smooth = signal.convolve(Y_int, filt_kernel, mode='same') / filt_len
        # Y_smooth = signal.medfilt(Y_int, filt_len)
        Y_rectified = Y_int - Y_smooth
        Y_rectified[Y_rectified < 0] = 0
        v = Y_rectified / np.max(Y_rectified)

    return v, F_coef_cents

def template_comb(M, theta=0):
    """Compute a comb template on a pitch axis

    Notebook: C3/C3S1_TranspositionTuning.ipynb

    Args:
        M (int): Length template (given in cents)
        theta (int): Shift parameter (given in cents); -50 <= theta < 50 (Default value = 0)

    Returns:
        template (np.ndarray): Comb template shifted by theta
    """
    template = np.zeros(M)
    peak_positions = (np.arange(0, M, 100) + theta)
    peak_positions = np.intersect1d(peak_positions, np.arange(M)).astype(int)
    template[peak_positions] = 1
    return template

def tuning_similarity(v):
    """Compute tuning similarity

    Notebook: C3/C3S1_TranspositionTuning.ipynb

    Args:
        v (np.ndarray): Vector representing an overall frequency distribution

    Returns:
        theta_axis (np.ndarray): Axis consisting of all tuning parameters -50 <= theta < 50
        sim (np.ndarray): Similarity values for all tuning parameters
        ind_max (int): Maximizing index
        theta_max (int): Maximizing tuning parameter
        template_max (np.ndarray): Similiarty-maximizing comb template
    """
    theta_axis = np.arange(-50, 50)  # Axis (given in cents)
    num_theta = len(theta_axis)
    sim = np.zeros(num_theta)
    M = len(v)
    for i in range(num_theta):
        theta = theta_axis[i]
        template = template_comb(M=M, theta=theta)
        sim[i] = np.inner(template, v)
    sim = sim / np.max(sim)
    ind_max = np.argmax(sim)
    theta_max = theta_axis[ind_max]
    template_max = template_comb(M=M, theta=theta_max)
    return theta_axis, sim, ind_max, theta_max, template_max

fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3_F08_C-major-scale.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, sr=Fs)

v, F_coef_cents = compute_freq_distribution(x, Fs=Fs, N=16384, gamma=10,
                                            local=True, filt_len=101)
theta_axis, sim, ind_max, theta_max, template_max = tuning_similarity(v)

print('Estimated tuning: %d cents' % theta_max)

# %% [markdown]
# ## Parameter Dependency
#
# To obtain a better idea on how this tuning estimation works, we visualize the
# similarity between v and t_theta as a function over theta.

# %%
def plot_tuning_similarity(sim, theta_axis, theta_max, ax=None, title=None, figsize=(4, 3)):
    """Plots tuning similarity

    Notebook: C3/C3S1_TranspositionTuning.ipynb

    Args:
        sim: Similarity values
        theta_axis: Axis consisting of cent values [-50:49]
        theta_max: Maximizing tuning parameter
        ax: Axis (in case of ax=None, figure is generated) (Default value = None)
        title: Title of figure (or subplot) (Default value = None)
        figsize: Size of figure (only used when ax=None) (Default value = (4, 3))

    Returns:
        fig: Handle for figure
        ax: Handle for axes
        line: handle for line plot
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
    if title is None:
        title = 'Estimated tuning: %d cents' % theta_max
    line = ax.plot(theta_axis, sim, 'k')
    ax.set_xlim([theta_axis[0], theta_axis[-1]])
    ax.set_ylim([0, 1.1])
    ax.plot([theta_max, theta_max], [0, 1.1], 'r')
    ax.set_xlabel('Tuning parameter (cents)')
    ax.set_ylabel('Similarity')
    ax.set_title(title)
    if fig is not None:
        plt.tight_layout()
    return fig, ax, line

def plot_freq_vector_template(v, F_coef_cents, template_max, theta_max, ax=None, title=None, figsize=(8, 3)):
    """Plots frequency distribution and similarity-maximizing template

    Notebook: C3/C3S1_TranspositionTuning.ipynb

    Args:
        v: Vector representing an overall frequency distribution
        F_coef_cents: Frequency axis
        template_max: Similarity-maximizing template
        theta_max: Maximizing tuning parameter
        ax: Axis (in case of ax=None, figure is generated) (Default value = None)
        title: Title of figure (or subplot) (Default value = None)
        figsize: Size of figure (only used when ax=None) (Default value = (8, 3))

    Returns:
        fig: Handle for figure
        ax: Handle for axes
        line: handle for line plot
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
    if title is None:
        title = r'Frequency distribution with maximizing comb template ($\theta$ = %d cents)' % theta_max
    line = ax.plot(F_coef_cents, v, c='k', linewidth=1)
    ax.set_xlim([F_coef_cents[0], F_coef_cents[-1]])
    ax.set_ylim([0, 1.1])
    x_ticks_freq = np.array([0, 1200, 2400, 3600, 4800, 6000, 7200, 8000])
    ax.plot(F_coef_cents, template_max * 1.1, 'r:', linewidth=0.5)
    ax.set_xticks(x_ticks_freq)
    ax.set_xlabel('Frequency (cents)')
    plt.title(title)
    if fig is not None:
        plt.tight_layout()
    return fig, ax, line

# Load audio signal
fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3_F08_C-major-scale.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, sr=Fs)

print('Average STFT (Local=True), without enhancement (Filt=False):', flush=True)
v, F_coef_cents = compute_freq_distribution(x, Fs=Fs, N=16384, gamma=10,
                                            local=True, filt=False)
theta_axis, sim, ind_max, theta_max, template_max = tuning_similarity(v)
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3],
                                          'height_ratios': [1]}, figsize=(12, 2))
plot_tuning_similarity(sim, theta_axis, theta_max, ax=ax[0])
plot_freq_vector_template(v, F_coef_cents, template_max, theta_max, ax=ax[1])
plt.show()

print('Average STFT (Local=True), with enhancement (Filt=True):', flush=True)
v, F_coef_cents = compute_freq_distribution(x, Fs=Fs, N=16384, gamma=10,
                                            local=True, filt_len=101)
theta_axis, sim, ind_max, theta_max, template_max = tuning_similarity(v)
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3],
                                          'height_ratios': [1]}, figsize=(12, 2))
plot_tuning_similarity(sim, theta_axis, theta_max, ax=ax[0])
plot_freq_vector_template(v, F_coef_cents, template_max, theta_max, ax=ax[1])
plt.show()

print('Global DFT (Local=False), with enhancement (Filt=True):', flush=True)
v, F_coef_cents = compute_freq_distribution(x, Fs=Fs, N=16384, gamma=10,
                                            local=False, filt_len=101)
theta_axis, sim, ind_max, theta_max, template_max = tuning_similarity(v)
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3],
                                          'height_ratios': [1]}, figsize=(12, 2))
plot_tuning_similarity(sim, theta_axis, theta_max, ax=ax[0])
plot_freq_vector_template(v, F_coef_cents, template_max, theta_max, ax=ax[1])
plt.show()

# %% [markdown]
# ## Music Examples

# %%
from IPython.display import Audio

fn_wav_dict = {}
fn_wav_dict['Cmaj'] = os.path.join('..', 'data', 'C3', 'FMP_C3_F08_C-major-scale.wav')
fn_wav_dict['Cmaj40'] = os.path.join('..', 'data', 'C3', 'FMP_C3_F08_C-major-scale_40-cents-up.wav')
fn_wav_dict['C4violin'] = os.path.join('..', 'data', 'C3', 'FMP_C3_NoteC4_Violin.wav')
fn_wav_dict['Burgmueller'] = os.path.join('..', 'data', 'C3', 'FMP_C3_F05_BurgmuellerFirstPart.wav')
fn_wav_dict['Weber'] = os.path.join('..', 'data', 'C8', 'FMP_C8_F10_Weber_Freischuetz-06_FreiDi-35-40.wav')

for name in fn_wav_dict:
    fn_wav = fn_wav_dict[name]
    Fs = 22050
    x, Fs = librosa.load(fn_wav, sr=Fs)
    print('Audio example: %s' % name)
    display( Audio(x, rate=Fs) )
    v, F_coef_cents = compute_freq_distribution(x, Fs=Fs, N=16384, gamma=10,
                                                local=True, filt_len=101)
    theta_axis, sim, ind_max, theta_max, template_max = tuning_similarity(v)
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3],
                                              'height_ratios': [1]}, figsize=(12, 2))
    plot_tuning_similarity(sim, theta_axis, theta_max, ax=ax[0])
    title = None
    plot_freq_vector_template(v, F_coef_cents, template_max, theta_max, ax=ax[1],  title=title)
    plt.show()

# %% [markdown]
# ## Dependency on Frequency Range
#
# Another important aspect for tuning estimation is the frequency range considered.

# %%
fn_wav = fn_wav_dict['Weber']
Fs = 22050
x, Fs = librosa.load(fn_wav, sr=Fs)

v, F_coef_cents = compute_freq_distribution(x, Fs=Fs, N=16384, gamma=10, local=True, filt_len=101)
theta_axis, sim, ind_max, theta_max, template_max = tuning_similarity(v)
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3],
                                          'height_ratios': [1]}, figsize=(12, 2))
plot_tuning_similarity(sim, theta_axis, theta_max, ax=ax[0])
plot_freq_vector_template(v, F_coef_cents, template_max, theta_max, ax=ax[1])
plt.show()

v, F_coef_cents = compute_freq_distribution(x, Fs=Fs, N=16384, gamma=10, local=True, filt_len=101)
v[6000:] = 0
theta_axis, sim, ind_max, theta_max, template_max = tuning_similarity(v)
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3],
                                          'height_ratios': [1]}, figsize=(12, 2))
plot_tuning_similarity(sim, theta_axis, theta_max, ax=ax[0])
plot_freq_vector_template(v, F_coef_cents, template_max, theta_max, ax=ax[1])
plt.show()

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

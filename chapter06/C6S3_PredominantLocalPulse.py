# %% [markdown]
# # Predominant Local Pulse (PLP)
#
# Following Section 6.3.1 of [Müller, FMP, Springer 2015], we introduce in this notebook an
# algorithm to compute predominant local pulse (PLP).

# %% [markdown]
# ## Main Idea
#
# The task of beat and pulse tracking can be seen as an extension of tempo estimation in the
# sense that, additionally to the rate, it also considers the phase of the pulses. The idea
# of the Fourier Tempogram was to locally compare a given novelty curve with windowed sinusoids.
# Based on this idea, we determine for each time position a windowed sinusoid that best captures
# the local peak structure of the novelty function. Instead of looking at the windowed sinusoids
# individually, the crucial idea is to employ an overlap-add technique by accumulating all
# sinusoids over time. As a result, one obtains a single function that can be regarded as a
# local periodicity enhancement of the original novelty function. Revealing **predominant local
# pulse** (PLP) information, this representation is referred to as a **PLP function**.

# %% [markdown]
# ## Optimal Windowed Sinusoids
#
# Given a novelty function $\Delta:\mathbb{Z}\to\mathbb{R}$, we derived the Fourier tempogram
# from the complex Fourier coefficient $\mathcal{F}(n,\omega)$.
#
# For each time position $n\in\mathbb{Z}$, we now consider the tempo parameter $\tau_n\in \Theta$
# that maximizes $\mathcal{T}^\mathrm{F}(n,\tau)$:
#
# $$\tau_n := \underset{\tau\in\Theta}{\mathrm{argmax}} \mathcal{T}^\mathrm{F}(n,\tau)$$
#
# The **phase information** encoded by the complex Fourier coefficient $\mathcal{F}(n,\omega)$
# can be used to derive the phase $\varphi_n$ of the windowed sinusoid of tempo $\tau_n$ that
# best correlates with the local section around $n$ of the novelty function $\Delta$.

# %% [markdown]
# ## Example: Shostakovich

# %%
import os
import sys
import numpy as np
from numba import jit
import librosa
from scipy import signal
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import IPython.display as ipd

sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c6

%matplotlib inline

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F07_Shostakovich_Waltz-02-Section_IncreasingTempo.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512,
                                                 gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

N = 500 #corresponding to 5 seconds (Fs_nov = 100 Hz)
H = 10  #Hopsize leading to a tempogram resolution of 10 Hz
Theta = np.arange(30, 601) #Tempo range parameter
X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs=Fs_nov, N=N, H=H,
                                                            Theta=Theta)
tempogram = np.abs(X)

t_nov = np.arange(nov.shape[0]) / Fs_nov
coef_n = np.array([0, 10, 20, 30, 40, 50, 60])
coef_k = np.zeros(len(coef_n), dtype=int)

for i in range(len(coef_n)):
    coef_k[i] = np.argmax(tempogram[:,coef_n[i]])

fig, ax, im = libfmp.b.plot_matrix(tempogram, T_coef=T_coef, F_coef=F_coef_BPM,
                                   figsize=(6.5, 3), ylabel='Tempo (BPM)',
                                   title='Fourier tempogram')
ax[0].plot(T_coef[coef_n], F_coef_BPM[coef_k], 'ro')

for i in range(len(coef_n)):
    n = coef_n[i]
    k = coef_k[i]
    tempo = F_coef_BPM[k]
    time = T_coef[n]
    corr = np.abs(X[k,n])
    kernel, t_kernel, t_kernel_sec = libfmp.c6.compute_sinusoid_optimal(X[k,n],
                                            F_coef_BPM[k], n, Fs_nov, N, H)
    title=r'Windowed sinusoid (t = %0.1f sec, $\tau$ = %0.0f BPM, corr = %0.2f)'% (time, tempo, corr)
    libfmp.c6.plot_signal_kernel(nov, t_nov, 0.5*kernel, t_kernel_sec,
                                 figsize=(6, 1.5), title=title)

# %% [markdown]
# ## Definition of PLP Function
#
# The estimation of optimal windowed sinusoids in regions with a strongly corrupt peak structure
# is problematic. To make the periodicity estimation more robust while keeping the temporal
# flexibility, the idea is to form a single function instead of looking at the sinusoids in a
# one-by-one fashion. To this end, we apply an **overlap–add** technique, where the optimal
# windowed sinusoids $\kappa_n$ are accumulated over all time positions $n\in\mathbb{Z}$.
# Furthermore, we only consider the positive part of the resulting function (half-wave
# rectification). The resulting function is our mid-level representation referred to as
# **PLP function**.

# %%
L = nov.shape[0]
N_left = N // 2
L_left = N_left
L_right = N_left
L_pad = L + L_left + L_right
t_pad = np.arange(L_pad)

t_nov = np.arange(nov.shape[0]) / Fs_nov
coef_n = np.array([0, 10, 20, 30, 40, 50, 60])
coef_k = np.zeros(len(coef_n), dtype=int)

for i in range(len(coef_n)):
    coef_k[i] = np.argmax(tempogram[:, coef_n[i]])

fig, ax = plt.subplots(5, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                          'height_ratios': [1, 2, 2, 1, 1]}, figsize=(8, 9))

libfmp.b.plot_signal(nov, Fs_nov, ax=ax[0, 0], color='k', title='Novelty function')
ax[0, 1].set_axis_off()

libfmp.b.plot_matrix(tempogram, T_coef=T_coef, F_coef=F_coef_BPM, ax=[ax[1,0], ax[1,1]],
                     title='Fourier tempogram', ylabel='Tempo (BPM)', colorbar=True)
ax[1, 0].plot(T_coef[coef_n], F_coef_BPM[coef_k], 'ro')

libfmp.b.plot_signal(nov, Fs_nov, ax=ax[2, 0], color='k', title='Novelty function with windowed sinusoids')
ax[2, 1].set_axis_off()

nov_PLP = np.zeros(L_pad)
for i in range(len(coef_n)):
    n = coef_n[i]
    k = coef_k[i]
    tempo = F_coef_BPM[k]
    time = T_coef[n]
    kernel, t_kernel, t_kernel_sec = libfmp.c6.compute_sinusoid_optimal(X[k, n], F_coef_BPM[k], n, Fs_nov, N, H)
    nov_PLP[t_kernel] = nov_PLP[t_kernel] + kernel
    ax[2, 0].plot(t_kernel_sec, 0.5 * kernel, 'r')
    ax[2, 0].set_ylim([-0.6, 1])

nov_PLP = nov_PLP[L_left:L_pad-L_right]
libfmp.b.plot_signal(nov_PLP, Fs_nov, ax=ax[3, 0], color='r', title='Accumulated windowed sinusoids')
ax[3,1].set_axis_off()

nov_PLP[nov_PLP < 0] = 0
libfmp.b.plot_signal(nov_PLP, Fs_nov, ax=ax[4, 0], color='r', title='PLP function $\Gamma$')
ax[4, 1].set_axis_off()

plt.tight_layout()

# %% [markdown]
# Note how the maxima of the different windowed sinusoids align well not only with the peaks
# of the novelty function, but also with the maxima of neighboring sinusoids in the overlapping
# regions, which leads to **constructive interferences**.

# %%
@jit(nopython=True)
def compute_plp(X, Fs, L, N, H, Theta):
    """Compute windowed sinusoid with optimal phase

    Notebook: C6/C6S3_PredominantLocalPulse.ipynb

    Args:
        X (np.ndarray): Fourier-based (complex-valued) tempogram
        Fs (scalar): Sampling rate
        L (int): Length of novelty curve
        N (int): Window length
        H (int): Hop size
        Theta (np.ndarray): Set of tempi (given in BPM)

    Returns:
        nov_PLP (np.ndarray): PLP function
    """
    win = np.hanning(N)
    N_left = N // 2
    L_left = N_left
    L_right = N_left
    L_pad = L + L_left + L_right
    nov_PLP = np.zeros(L_pad)
    M = X.shape[1]
    tempogram = np.abs(X)
    for n in range(M):
        k = np.argmax(tempogram[:, n])
        tempo = Theta[k]
        omega = (tempo / 60) / Fs
        c = X[k, n]
        phase = - np.angle(c) / (2 * np.pi)
        t_0 = n * H
        t_1 = t_0 + N
        t_kernel = np.arange(t_0, t_1)
        kernel = win * np.cos(2 * np.pi * (t_kernel * omega - phase))
        nov_PLP[t_kernel] = nov_PLP[t_kernel] + kernel
    nov_PLP = nov_PLP[L_left:L_pad-L_right]
    nov_PLP[nov_PLP < 0] = 0
    return nov_PLP

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F07_Shostakovich_Waltz-02-Section_IncreasingTempo.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512,
                                                 gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

L = len(nov)
N = 500
H = 10
Theta = np.arange(30, 601)
X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs=Fs_nov, N=N, H=H,
                                                            Theta=Theta)
nov_PLP = compute_plp(X, Fs_nov, L, N, H, Theta)

t_nov = np.arange(nov.shape[0]) / Fs_nov
peaks, properties = signal.find_peaks(nov, prominence=0.02)
peaks_sec = t_nov[peaks]
libfmp.b.plot_signal(nov, Fs_nov, color='k', title='Novelty function with detected peaks');
plt.plot(peaks_sec, nov[peaks], 'ro')
plt.show()
x_peaks = librosa.clicks(peaks_sec, sr=Fs, click_freq=1000, length=len(x))
ipd.display(ipd.Audio(x + x_peaks, rate=Fs))

peaks, properties = signal.find_peaks(nov_PLP, prominence=0.02)
peaks_sec = t_nov[peaks]
libfmp.b.plot_signal(nov_PLP, Fs_nov, color='k', title='PLP function with detected peaks');
plt.plot(peaks_sec, nov_PLP[peaks], 'ro')
plt.show()
x_peaks = librosa.clicks(peaks_sec, sr=Fs, click_freq=1000, length=len(x))
ipd.display(ipd.Audio(x + x_peaks, rate=Fs))

# %% [markdown]
# ## Example: Brahms
#
# As a second example, we consider an orchestra recording of the Hungarian Dance No. 5 by
# Johannes Brahms. This excerpt is challenging because of several sudden tempo changes.
# Furthermore, the input novelty function is very noisy due to many soft and fuzzy onsets.

# %%
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F19_Brahms_Ormandy_sec35-53.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512,
                                                 gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

N = 500
H = 10
Theta = np.arange(30, 601)
L = len(nov)
X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs=Fs_nov, N=N, H=H,
                                                            Theta=Theta)
nov_PLP = compute_plp(X, Fs_nov, L, N, H, Theta)

tempogram = np.abs(X)
title = 'Fourier tempogram using a window length of %0.1f seconds'%(N/Fs_nov)
libfmp.b.plot_matrix(tempogram, T_coef=T_coef, F_coef=F_coef_BPM, figsize=(8,2.5),
                     title=title, ylabel='Tempo (BPM)', colorbar=True);
fig, ax, im = libfmp.b.plot_matrix(tempogram, T_coef=T_coef, F_coef=F_coef_BPM,
                                   figsize=(8,2.5), ylabel='Tempo (BPM)', colorbar=True,
                                   title='Fourier tempogram with dominant local tempo')
coef_k = np.argmax(tempogram, axis=0)
ax[0].plot(T_coef, F_coef_BPM[coef_k], 'r.')

t_nov = np.arange(nov.shape[0]) / Fs_nov
peaks, properties = signal.find_peaks(nov, prominence=0.05)
peaks_sec = t_nov[peaks]
libfmp.b.plot_signal(nov, Fs_nov, color='k', figsize=(7,2),
                     title='Novelty function with detected peaks');
plt.plot(peaks_sec, nov[peaks], 'r.')
plt.show()
x_peaks = librosa.clicks(peaks_sec, sr=Fs, click_freq=1000, length=len(x))
ipd.display(ipd.Audio(x + x_peaks, rate=Fs))

peaks, properties = signal.find_peaks(nov_PLP, prominence=0.05)
peaks_sec = t_nov[peaks]
libfmp.b.plot_signal(nov_PLP, Fs_nov, color='k', figsize=(7,2),
                     title='PLP function with detected peaks');
plt.plot(peaks_sec, nov_PLP[peaks], 'ro')
plt.show()
x_peaks = librosa.clicks(peaks_sec, sr=Fs, click_freq=1000, length=len(x))
ipd.display(ipd.Audio(x + x_peaks, rate=Fs))

# %% [markdown]
# ## Dependency of Sinusoid Window Length
#
# When computing a PLP function, there are many parameters involved. First of all, the final
# result depends on input novelty curve, which can be computed in many different ways. In this
# notebook, we have a closer look at the length of the windowed sinusoids used to compute the
# Fourier tempogram. Obviously, one has a trade-off between contradicting principles. Using
# longer windows, one obtains more accurate and robust tempo estimates, but looses temporal
# flexibility. In contrast, using shorter windows one increases the temporal resolution, but
# makes the tempo estimates more vulnerable.

# %%
def compute_plot_tempogram_plp(fn_wav, Fs=22050, N=500, H=10, Theta=np.arange(30, 601),
                               title='', figsize=(8, 4), plot_maxtempo=False):
    """Compute and plot Fourier-based tempogram and PLP function

    Notebook: C6/C6S3_PredominantLocalPulse.ipynb

    Args:
        fn_wav: Filename of audio file
        Fs: Sample rate (Default value = 22050)
        N: Window size (Default value = 500)
        H: Hop size (Default value = 10)
        Theta: Set of tempi (given in BPM) (Default value = np.arange(30, 601))
        title: Title of figure (Default value = '')
        figsize: Figure size (Default value = (8, 4))
        plot_maxtempo: Visualize tempo with greatest coefficients in tempogram (Default value = False)
    """
    x, Fs = librosa.load(fn_wav, Fs)

    nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512, gamma=100, M=10, norm=True)
    nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

    L = len(nov)
    H = 10
    X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs=Fs_nov, N=N, H=H, Theta=Theta)
    nov_PLP = compute_plp(X, Fs_nov, L, N, H, Theta)
    tempogram = np.abs(X)

    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                              'height_ratios': [2, 1]},
                           figsize=figsize)
    libfmp.b.plot_matrix(tempogram, T_coef=T_coef, F_coef=F_coef_BPM, title=title,
                         ax=[ax[0, 0], ax[0, 1]], ylabel='Tempo (BPM)', colorbar=True)
    if plot_maxtempo:
        coef_k = np.argmax(tempogram, axis=0)
        ax[0, 0].plot(T_coef, F_coef_BPM[coef_k], 'r.')

    t_nov = np.arange(nov.shape[0]) / Fs_nov
    peaks, properties = signal.find_peaks(nov_PLP, prominence=0.05)
    peaks_sec = t_nov[peaks]
    libfmp.b.plot_signal(nov_PLP, Fs_nov, color='k', ax=ax[1, 0])
    ax[1, 1].set_axis_off()
    ax[1, 0].plot(peaks_sec, nov_PLP[peaks], 'ro')
    plt.show()
    x_peaks = librosa.clicks(peaks_sec, sr=Fs, click_freq=1000, length=len(x))
    ipd.display(ipd.Audio(x + x_peaks, rate=Fs))

N=1200
title='Fourier tempogram using a window length of %0.1f seconds'%(N/Fs_nov)
compute_plot_tempogram_plp(fn_wav, N=N, title=title, plot_maxtempo=True)

N=200
title='Fourier tempogram using a window length of %0.1f seconds'%(N/Fs_nov)
compute_plot_tempogram_plp(fn_wav, N=N, title=title, plot_maxtempo=True)

# %% [markdown]
# ## Dependency of Tempo Set
#
# Taking the frame-wise maximum to determine the predominant tempo has both advantages and
# drawbacks. On the one hand, it allows the PLP function to quickly adjust—even to sudden
# changes in tempo. On the other hand, it may lead to unwanted jumps such as random switches
# between tempo octaves. Such switches in the pulse level can be avoided when constraining
# the tempo set $\Theta$ in the maximization.

# %%
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F20_Burgmueller_Op100-02-FirstPart.wav')

title=r'Fourier tempogram and PLP function with $\Theta=[30:600]$'
compute_plot_tempogram_plp(fn_wav, N=500, Theta=np.arange(30, 601), title=title)

title=r'Fourier tempogram and PLP function with $\Theta=[60:200]$'
compute_plot_tempogram_plp(fn_wav, N=500, Theta=np.arange(60,201), title=title)

title=r'Fourier tempogram and PLP function with $\Theta=[200:340]$'
compute_plot_tempogram_plp(fn_wav, N=500, Theta=np.arange(200,341), title=title)

title=r'Fourier tempogram and PLP function with $\Theta=[450:600]$'
compute_plot_tempogram_plp(fn_wav, N=500, Theta=np.arange(450,601), title=title)

# %% [markdown]
# ## Further Notes
#
# The detection of locally periodic patterns becomes challenging when the music recording
# reveals significant tempo changes. This often occurs in performances of classical music as
# a result of ritardandi, accelerandi, fermate, and so on. In particular, for highly expressive
# performances (e.g., interpretations of romantic piano music), the assumption of local
# quasiperiodicity is often violated, leading to poor beat tracking result. Furthermore, the
# notions of tempo and beat are often vague and subjective, due to the complex hierarchical
# structure of rhythm. The PLP concept introduced in this notebook does not explicitly address
# the problem of extracting pulses at a specific level. Instead, a **PLP function** can be
# regarded as a kind of **mid-level representation** that captures the locally predominant pulse.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

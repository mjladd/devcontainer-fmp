# %% [markdown]
# # Fourier Tempogram
#
# Following Section 6.2.2 of [Müller, FMP, Springer 2015], we show in this notebook how to
# compute a tempogram using a variant of a short-time Fourier transform.

# %% [markdown]
# ## Definition
#
# We assume that we are given a discrete-time novelty function $\Delta:\mathbb{Z}\to\mathbb{R}$
# in which peaks indicate note onset candidates. The idea of Fourier analysis is to detect
# local periodicities in novelty curve by comparing it with windowed sinusoids. A high
# correlation of a local section of $\Delta$ with a windowed sinusoid indicates a periodicity
# of the sinusoid's frequency (given a suitable phase). This correlation (along with the phase)
# can be computed using a short-time Fourier transform.
#
# For a frequency parameter $\omega\in\mathbb{R}_{\geq 0}$ and time parameter $n\in\mathbb{Z}$,
# the complex Fourier coefficient $\mathcal{F}(n,\omega)$ is defined by
#
# $$\mathcal{F}(n,\omega) := \sum_{m\in\mathbb{Z}} \Delta(m)\overline{w}(m-n)\mathrm{exp}(-2\pi i\omega m)$$
#
# Converting frequency to tempo values, we define the (discrete) **Fourier tempogram**
# $\mathcal{T}^\mathrm{F}: \mathbb{Z} \times \Theta \to \mathbb{R}_{\geq 0}$ by
#
# $$\mathcal{T}^\mathrm{F}(n,\tau) := |\mathcal{F}(n,\tau/60)|$$

# %% [markdown]
# ## Tempo Resolution
#
# Using a tempo set like $\Theta=[30:600]$ requires a spectral analysis of high resolution—in
# particular in the lower frequency range. A straightforward STFT based on the DFT may not be
# suitable. One alternative is to compute the required Fourier coefficients individually
# (without using the DFT). Even though this cannot be done via the FFT algorithm, the
# computational complexity may still be reasonable since only a relatively small number of
# Fourier coefficients (corresponding to the tempo set $\Theta$) need to be computed.

# %% [markdown]
# ## Implementation

# %%
import os, sys
import numpy as np
import librosa
from scipy import signal
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import IPython.display as ipd
from numba import jit

sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c3
import libfmp.c6

%matplotlib inline


@jit(nopython=True)
def compute_tempogram_fourier(x, Fs, N, H, Theta=np.arange(30, 601, 1)):
    """Compute Fourier-based tempogram [FMP, Section 6.2.2]

    Notebook: C6/C6S2_TempogramFourier.ipynb

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size
        Theta (np.ndarray): Set of tempi (given in BPM) (Default value = np.arange(30, 601, 1))

    Returns:
        X (np.ndarray): Tempogram
        T_coef (np.ndarray): Time axis (seconds)
        F_coef_BPM (np.ndarray): Tempo axis (BPM)
    """
    win = np.hanning(N)
    N_left = N // 2
    L = x.shape[0]
    L_left = N_left
    L_right = N_left
    L_pad = L + L_left + L_right
    # x_pad = np.pad(x, (L_left, L_right), 'constant')  # doesn't work with jit
    x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
    t_pad = np.arange(L_pad)
    M = int(np.floor(L_pad - N) / H) + 1
    K = len(Theta)
    X = np.zeros((K, M), dtype=np.complex_)

    for k in range(K):
        omega = (Theta[k] / 60) / Fs
        exponential = np.exp(-2 * np.pi * 1j * omega * t_pad)
        x_exp = x_pad * exponential
        for n in range(M):
            t_0 = n * H
            t_1 = t_0 + N
            X[k, n] = np.sum(win * x_exp[t_0:t_1])
        T_coef = np.arange(M) * H / Fs
        F_coef_BPM = Theta
    return X, T_coef, F_coef_BPM

Fs = 100
L = 10*Fs
x = np.zeros(L)
peaks = np.concatenate((np.arange(40,L//2,40),np.arange(500,L,50)))
x[peaks]=1

N = 300 #corresponding to 3 seconds (Fs = 100)
H = 10
Theta = np.arange(50,410,10)

X, T_coef, F_coef_BPM = compute_tempogram_fourier(x, Fs, N=N, H=H, Theta=Theta)
tempogram = np.abs(X)

fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                          'height_ratios': [1, 2]}, figsize=(8, 5))
libfmp.b.plot_signal(x, Fs, ax=ax[0,0], color='k', title='Novelty function')
ax[0,1].set_axis_off()
libfmp.b.plot_matrix(tempogram, T_coef=T_coef, F_coef=F_coef_BPM, ax=[ax[1,0], ax[1,1]],
                     title='Fourier tempogram', ylabel='Tempo (BPM)', colorbar=True);
plt.tight_layout()

# %% [markdown]
# ## Interpretation
#
# The visualization of the Fourier tempogram $\mathcal{T}^\mathrm{F}$ reveals the dominant tempo
# over time. Being based on a Fourier analysis, an entry $\mathcal{T}^\mathrm{F}(n,\tau)$ is
# obtained by locally comparing the novelty function $\Delta$ in a neighborhood of $n$ with a
# windowed sinusoid that represents the tempo $\tau$ (or the frequency $\omega=\tau/60$).

# %%
def compute_sinusoid_optimal(c, tempo, n, Fs, N, H):
    """Compute windowed sinusoid with optimal phase

    Notebook: C6/C6S2_TempogramFourier.ipynb

    Args:
        c (complex): Coefficient of tempogram (c=X(k,n))
        tempo (float): Tempo parameter corresponding to c (tempo=F_coef_BPM[k])
        n (int): Frame parameter of c
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size

    Returns:
        kernel (np.ndarray): Windowed sinusoid
        t_kernel (np.ndarray): Time axis (samples) of kernel
        t_kernel_sec (np.ndarray): Time axis (seconds) of kernel
    """
    win = np.hanning(N)
    N_left = N // 2
    omega = (tempo / 60) / Fs
    t_0 = n * H
    t_1 = t_0 + N
    phase = - np.angle(c) / (2 * np.pi)
    t_kernel = np.arange(t_0, t_1)
    kernel = win * np.cos(2 * np.pi * (t_kernel*omega - phase))
    t_kernel_sec = (t_kernel - N_left) / Fs
    return kernel, t_kernel, t_kernel_sec

def plot_signal_kernel(x, t_x, kernel, t_kernel, xlim=None, figsize=(8, 2), title=None):
    """Visualize signal and local kernel

    Notebook: C6/C6S2_TempogramFourier.ipynb

    Args:
        x: Signal
        t_x: Time axis of x (given in seconds)
        kernel: Local kernel
        t_kernel: Time axis of kernel (given in seconds)
        xlim: Limits for x-axis (Default value = None)
        figsize: Figure size (Default value = (8, 2))
        title: Title of figure (Default value = None)

    Returns:
        fig: Matplotlib figure handle
    """
    if xlim is None:
        xlim = [t_x[0], t_x[-1]]
    fig = plt.figure(figsize=figsize)
    plt.plot(t_x, x, 'k')
    plt.plot(t_kernel, kernel, 'r')
    plt.title(title)
    plt.xlim(xlim)
    plt.tight_layout()
    return fig

t_x = np.arange(x.shape[0])/Fs
coef_n = [20, 30, 70]
coef_k = [11, 5, 19]

fig, ax, im = libfmp.b.plot_matrix(tempogram,
                T_coef=T_coef, F_coef=F_coef_BPM, figsize=(9,3),
                title='Fourier tempogram', ylabel='Tempo (BPM)', colorbar=True);
ax[0].plot(T_coef[coef_n],F_coef_BPM[coef_k],'ro')

for i in range(len(coef_k)):
    k = coef_k[i]
    n = coef_n[i]
    tempo = F_coef_BPM[k]
    time = T_coef[n]
    corr = np.abs(X[k,n])
    kernel, t_kernel, t_kernel_sec = compute_sinusoid_optimal(X[k,n],
                        F_coef_BPM[k], n, Fs, N, H)
    title=r'Windowed sinusoid (t = %0.1f sec, $\tau$ = %0.0f BPM, corr = %0.2f)'%(time, tempo, corr)
    fig = plot_signal_kernel(x, t_x, kernel, t_kernel_sec, title=title)
plt.tight_layout()

# %% [markdown]
# For the first time-tempo pair, the positive parts of the windowed sinusoid nicely align with
# the impulse-like peaks of the novelty function $\Delta$, whereas the negative parts of the
# sinusoid fall into the zero-regions of $\Delta$. As a result, there is a high correlation
# between the windowed sinusoid and $\Delta$, which leads to a large coefficient.
#
# This discussion shows that a Fourier tempogram generally indicates **tempo harmonics**, but
# suppresses **tempo subharmonics**.

# %% [markdown]
# ## Example: Shostakovich

# %%
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F07_Shostakovich_Waltz-02-Section_IncreasingTempo.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512, gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

N = 500 #corresponding to 5 seconds (Fs_nov = 100)
H = 10
Theta = np.arange(30, 601)
X, T_coef, F_coef_BPM = compute_tempogram_fourier(nov, Fs_nov, N=N, H=H, Theta=Theta)
tempogram = np.abs(X)

fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                          'height_ratios': [1, 2]}, figsize=(8,5))
libfmp.b.plot_signal(nov, Fs_nov, ax=ax[0,0], color='k', title='Novelty function')
ax[0,1].set_axis_off()
libfmp.b.plot_matrix(tempogram, T_coef=T_coef, F_coef=F_coef_BPM, ax=[ax[1,0], ax[1,1]],
                     title='Fourier tempogram', ylabel='Tempo (BPM)', colorbar=True);
plt.tight_layout()

# %% [markdown]
# As the Fourier tempogram $\mathcal{T}^\mathrm{F}$ reveals, the dominant tempo of this excerpt
# is between $200$ and $300~\mathrm{BPM}$. Starting with roughly $\tau=225~\mathrm{BPM}$, the
# tempo slightly increases over time. Interestingly, because of the weak downbeats every third
# beat within the 3/4 meter, the tempogram $\mathcal{T}^\mathrm{F}$ also shows some larger
# coefficients that correspond to $1/3$ and $2/3$ of the main tempo.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

# %% [markdown]
# # Autocorrelation Tempogram
#
# Following Section 6.2.3 of [Müller, FMP, Springer 2015], we show in this notebook how to
# compute a tempogram using a short-time autocorrelation method.

# %% [markdown]
# ## Autocorrelation
#
# As a second periodicity estimation method, besides the Fourier-based approach, we now discuss
# an autocorrelation-based approach. Generally speaking, the **autocorrelation** is a
# mathematical tool for measuring the similarity of a signal with a time-shifted version of
# itself. Let $x:\mathbb{Z}\to\mathbb{R}$ be such discrete-time signal having finite energy.
# The autocorrelation $R_{xx}:\mathbb{Z}\to\mathbb{R}$ of the real-valued signal $x$ is defined by
#
# $$R_{xx}(\ell) = \sum_{m\in\mathbb{Z}} x(m)x(m-\ell)$$
#
# which yields a function that depends on the time-shift or **lag** parameter $\ell\in\mathbb{Z}$.
# The autocorrelation $R_{xx}(\ell)$ is maximal for $\ell=0$ and symmetric in $\ell$. Intuitively,
# if the autocorrelation is large for a given lag, then the signal contains repeating patterns
# that are separated by a time period as specified by the lag parameter.

# %%
import os, sys
import librosa
import numpy as np
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

N = 100
x = np.zeros(N)
x[np.arange(1,N,14)] = 1
x[np.arange(3,N,9)] = 1

fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 1.5, 1.5]}, figsize=(6, 5))

ax[0].plot(x, 'k')
ax[0].set_title('Signal $x$')
ax[0].set_xlabel('Time (samples)')

r_xx = np.correlate(x, x, mode='full')
lag_axis = np.arange(-(N-1),N)
ax[1].plot(lag_axis,r_xx, 'r')
ax[1].set_title('Autocorrelation (full mode)')
ax[1].set_xlabel(r'Lag parameter (samples)')

r_xx = np.correlate(x, x, mode='full')
lag_axis = np.arange(-(N-1),N)
ax[2].plot(lag_axis,r_xx, 'r')
ax[2].set_title('Autocorrelation (full mode)')
ax[2].set_xlabel(r'Lag parameter (samples)')
ax[2].set_xlim([0,50])
ax[2].set_ylim([0,11])

plt.tight_layout()

# %% [markdown]
# ## Short-Time Autocorrelation
#
# We now apply the autocorrelation in a local fashion for analyzing a given novelty function
# $\Delta:\mathbb{Z}\to\mathbb{R}$ in the neighborhood of a given time parameter $n$. To this
# end, we fix a window function $w:\mathbb{Z}\to\mathbb{R}$. To obtain the **short-time
# autocorrelation** $\mathcal{A}:\mathbb{Z}\times\mathbb{Z}\to\mathbb{R}$, we compute the
# autocorrelation of the windowed version of $\Delta$.

# %% [markdown]
# ## Implementation

# %%
# @jit(nopython=True)  # not possible because of np.correlate with mode='full'
def compute_autocorrelation_local(x, Fs, N, H, norm_sum=True):
    """Compute local autocorrelation [FMP, Section 6.2.3]

    Notebook: C6/C6S2_TempogramAutocorrelation.ipynb

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size
        norm_sum (bool): Normalizes by the number of summands in local autocorrelation (Default value = True)

    Returns:
        A (np.ndarray): Time-lag representation
        T_coef (np.ndarray): Time axis (seconds)
        F_coef_lag (np.ndarray): Lag axis
    """
    # L = len(x)
    L_left = round(N / 2)
    L_right = L_left
    x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
    L_pad = len(x_pad)
    M = int(np.floor(L_pad - N) / H) + 1
    A = np.zeros((N, M))
    win = np.ones(N)
    if norm_sum is True:
        lag_summand_num = np.arange(N, 0, -1)
    for n in range(M):
        t_0 = n * H
        t_1 = t_0 + N
        x_local = win * x_pad[t_0:t_1]
        r_xx = np.correlate(x_local, x_local, mode='full')
        r_xx = r_xx[N-1:]
        if norm_sum is True:
            r_xx = r_xx / lag_summand_num
        A[:, n] = r_xx
    Fs_A = Fs / H
    T_coef = np.arange(A.shape[1]) / Fs_A
    F_coef_lag = np.arange(N) / Fs
    return A, T_coef, F_coef_lag

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F07_Shostakovich_Waltz-02-Section_IncreasingTempo.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512, gamma=100, M=10, norm=1)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

#Autocorrelation parameters
N = 300 #corresponding to 3 seconds (Fs_nov = 100 Hz)
H = 10

A, T_coef, F_coef_lag = compute_autocorrelation_local(nov, Fs_nov, N, H, norm_sum=False)
A_norm, T_coef, F_coef_lag = compute_autocorrelation_local(nov, Fs_nov, N, H)

fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                          'height_ratios': [1, 2, 2]}, figsize=(8,7))

libfmp.b.plot_signal(nov, Fs_nov, ax=ax[0,0], color='k', title='Novelty function')
ax[0,1].set_axis_off()

libfmp.b.plot_matrix(A, ax=[ax[1,0], ax[1,1]], T_coef=T_coef, F_coef=F_coef_lag,
                     title=r'Time-lag representation $\mathcal{A}$', ylabel='Lag (seconds)', colorbar=True);

libfmp.b.plot_matrix(A_norm, ax=[ax[2,0], ax[2,1]], T_coef=T_coef, F_coef=F_coef_lag,
                     title=r'Time-lag representation $\mathcal{A}$ (with sum normalization)',
                     ylabel='Lag (seconds)', colorbar=True);
plt.tight_layout()

# %% [markdown]
# ## Interpretation

# %%
def plot_signal_local_lag(x, t_x, local_lag, t_local_lag, lag, xlim=None, figsize=(8, 1.5), title=''):
    """Visualize signal and local lag [FMP, Figure 6.14]

    Notebook: C6/C6S2_TempogramAutocorrelation.ipynb

    Args:
        x: Signal
        t_x: Time axis of x (given in seconds)
        local_lag: Local lag
        t_local_lag: Time axis of kernel (given in seconds)
        lag: Lag (given in seconds)
        xlim: Limits for x-axis (Default value = None)
        figsize: Figure size (Default value = (8, 1.5))
        title: Title of figure (Default value = '')

    Returns:
        fig: Matplotlib figure handle
    """
    if xlim is None:
        xlim = [t_x[0], t_x[-1]]
    fig = plt.figure(figsize=figsize)
    plt.plot(t_x, x, 'k:', linewidth=0.5)
    plt.plot(t_local_lag, local_lag, 'k', linewidth=3.0)
    plt.plot(t_local_lag+lag, local_lag, 'r', linewidth=2)
    plt.title(title)
    plt.ylim([0, 1.1 * np.max(x)])
    plt.xlim(xlim)
    plt.tight_layout()
    return fig

time_sec = np.array([2, 2, 2])
lag_sec = np.array([0.13, 0.26, 0.78])
coef_n = (time_sec * Fs_nov/H).astype(int)
coef_k = (lag_sec * Fs_nov).astype(int)

fig, ax, im = libfmp.b.plot_matrix(A, T_coef=T_coef, F_coef=F_coef_lag, figsize=(9,3),
                     title=r'Time-lag representation $\mathcal{A}$', ylabel='Lag (seconds)', colorbar=True);
ax[0].plot(T_coef[coef_n], F_coef_lag[coef_k],'ro')

L = len(nov)
L_left = round(N/2)
L_right = L_left
nov_pad = np.concatenate( ( np.zeros(L_left), nov, np.zeros(L_right) ) )
L_pad = len(nov_pad)
win = np.ones(N)

time_sec = np.array([2, 2, 2, 2])
lag_sec = np.array([0, 0.13, 0.26, 0.78])
t_nov = np.arange(nov.shape[0])/Fs_nov

for i in range(len(time_sec)):
    t_0 = time_sec[i] * Fs_nov
    t_1 = t_0 + N
    nov_local = win*nov_pad[t_0:t_1]
    t_nov_local = (np.arange(t_0,t_1) - L_left)/Fs_nov
    lag = lag_sec[i]
    title=r'Shifted novelty function (t = %0.2f sec, $\ell$ = %0.2f sec)'%(time_sec[i], lag)
    fig = plot_signal_local_lag(nov, t_nov, nov_local, t_nov_local, lag, title=title)

# %% [markdown]
# ## Autocorrelation Tempogram
#
# To obtain a time–tempo representation from the time–lag representation, one needs to convert
# the lag parameter into a tempo parameter. The tempo is given by
#
# $$\tau = \frac{60}{r\cdot \ell}~\mathrm{BPM}$$
#
# where $r$ is the time resolution and $\ell$ is the lag in frames.

# %%
# @jit(nopython=True)
def compute_tempogram_autocorr(x, Fs, N, H, norm_sum=False, Theta=np.arange(30, 601)):
    """Compute autocorrelation-based tempogram

    Notebook: C6/C6S2_TempogramAutocorrelation.ipynb

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size
        norm_sum (bool): Normalizes by the number of summands in local autocorrelation (Default value = False)
        Theta (np.ndarray): Set of tempi (given in BPM) (Default value = np.arange(30, 601))

    Returns:
        tempogram (np.ndarray): Tempogram tempogram
        T_coef (np.ndarray): Time axis T_coef (seconds)
        F_coef_BPM (np.ndarray): Tempo axis F_coef_BPM (BPM)
        A_cut (np.ndarray): Time-lag representation A_cut (cut according to Theta)
        F_coef_lag_cut (np.ndarray): Lag axis F_coef_lag_cut
    """
    tempo_min = Theta[0]
    tempo_max = Theta[-1]
    lag_min = int(np.ceil(Fs * 60 / tempo_max))
    lag_max = int(np.ceil(Fs * 60 / tempo_min))
    A, T_coef, F_coef_lag = compute_autocorrelation_local(x, Fs, N, H, norm_sum=norm_sum)
    A_cut = A[lag_min:lag_max+1, :]
    F_coef_lag_cut = F_coef_lag[lag_min:lag_max+1]
    F_coef_BPM_cut = 60 / F_coef_lag_cut
    F_coef_BPM = Theta
    tempogram = interp1d(F_coef_BPM_cut, A_cut, kind='linear',
                         axis=0, fill_value='extrapolate')(F_coef_BPM)
    return tempogram, T_coef, F_coef_BPM, A_cut, F_coef_lag_cut

Theta = np.arange(30, 601)
tempogram, T_coef, F_coef, A, F_coef_lag = compute_tempogram_autocorr(nov,
                            Fs_nov, N, H, norm_sum=False, Theta=Theta)

fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]}, figsize=(10,3))

libfmp.b.plot_matrix(A, T_coef=T_coef, F_coef=F_coef_lag, ax=[ax[0]],
                     title='Lag axis', ylabel='Lag (seconds)', colorbar=True);
lag_yticks = np.array([F_coef_lag[0], 0.5, 1.0, 1.5, F_coef_lag[-1]])
ax[0].set_yticks(lag_yticks)

libfmp.b.plot_matrix(A, T_coef=T_coef, F_coef=F_coef_lag, ax=[ax[1]],
                     title='Lag converted to BPM', ylabel='Beat (BPM)', colorbar=True);
ax[1].set_yticks(lag_yticks)
ax[1].set_yticklabels(60/lag_yticks)

libfmp.b.plot_matrix(tempogram, T_coef=T_coef, F_coef=F_coef, ax=[ax[2]],
                     title='Linear BPM axis', ylabel='Beat (BPM)', colorbar=True);
ax[2].set_yticks([F_coef[0], 120, 240, 360, 480, F_coef[-1]]);
plt.tight_layout()

# %% [markdown]
# ## Tempo Harmonics and Subharmonics
#
# An **autocorrelation tempogram** exhibits **tempo subharmonics**, while suppressing tempo
# harmonics. The reason for this behavior is that a high correlation of a local section of the
# novelty function with the section shifted by $\ell$ samples also implies a high correlation
# with a section shifted by $k\cdot\ell$ lags for integers $k\in\mathbb{N}$. Note that this
# behavior stands in contrast to the **Fourier tempogram**, which emphasizes **tempo harmonics**,
# but suppresses tempo subharmonics.

# %%
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F11_ClickTrack-BPM170-200.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512, gamma=100, M=10, norm=1)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

#Autocorrelation parameters
N = 500 #corresponding to 5 seconds (Fs_nov = 100 Hz)
H = 10
Theta = np.arange(30, 601)
tempogram_A, T_coef, F_coef, A, F_coef_lag = compute_tempogram_autocorr(nov, Fs_nov, N, H,
                                                 norm_sum=False, Theta=Theta)

fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                          'height_ratios': [1, 2, 2]}, figsize=(8,7))
libfmp.b.plot_signal(nov, Fs_nov, ax=ax[0,0], color='k', title='Novelty function')
ax[0,1].set_axis_off()
libfmp.b.plot_matrix(tempogram_A, T_coef=T_coef, F_coef=F_coef, ax=[ax[1,0], ax[1,1]],
                     title='Autocorrelation tempogram', ylabel='Tempo (BPM)', colorbar=True);


X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs_nov, N=N, H=H, Theta=Theta)
tempogram_F = np.abs(X)

libfmp.b.plot_matrix(tempogram_F, T_coef=T_coef, F_coef=F_coef_BPM, ax=[ax[2,0], ax[2,1]],
                     title='Fourier tempogram', ylabel='Tempo (BPM)', colorbar=True)
plt.tight_layout()

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Angel Villar-Corrales.

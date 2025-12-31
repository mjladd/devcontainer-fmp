# %% [markdown]
# # Adaptive Windowing
#
# Following Section 6.3.3 of [Müller, FMP, Springer 2015], we discuss in this notebook the
# concept of adaptive windowing.

# %% [markdown]
# ## Basic Idea
#
# One crucial step in practically all music analysis tasks consists of transforming the given
# audio signal into a suitable feature representation that captures certain musical properties.
# Since most musical properties vary over time, the given audio signal is typically split up
# into segments or frames, which are then further processed individually. The underlying
# assumption is that the signal stays (approximately) stationary within each segment with
# regard to the property to be captured.
#
# Using **fixed-size windowing**, however, may lead to a violation of the homogeneity assumption:
# the boundaries of the resulting windowed sections often do not coincide with the positions
# where the changes of the signal occur. As an alternative to fixed-size windowing, one can
# employ a more musically meaningful **adaptive windowing** strategy, where segment boundaries
# are induced by previously extracted onset and beat positions. Since musical changes typically
# occur at onset positions, this often leads to an increased homogeneity within the adaptively
# determined frames and a significant improvement in the resulting feature quality.

# %% [markdown]
# ## Example: C-Major Scale

# %%
import numpy as np
import os, sys, librosa
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import IPython.display as ipd

sys.path.append('..')
import libfmp.b
import libfmp.c6
%matplotlib inline

def plot_beat_grid(B_sec, ax, color='r', linestyle=':', linewidth=1):
    """Plot beat grid (given in seconds) into axis

    Notebook: C6/C6S3_AdaptiveWindowing.ipynb

    Args:
        B_sec: Beat grid
        ax: Axes for plotting
        color: Color of lines (Default value = 'r')
        linestyle: Style of lines (Default value = ':')
        linewidth: Width of lines (Default value = 1)
    """
    for b in B_sec:
        ax.axvline(x=b, color=color, linestyle=linestyle, linewidth=linewidth)

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F24_ScaleMajorC-middle.wav')

Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)

N, H = 1024, 512
X = librosa.stft(y=x, n_fft=N, hop_length=H, win_length=N, center=True, window='hanning')
Y = np.abs(X)
C = librosa.feature.chroma_stft(S=Y, sr=Fs, norm=1, hop_length=H, n_fft=N)

B_num = 10
B_fix_sec = np.linspace(0, x.shape[0], B_num) / Fs
B_fix_frame = (B_fix_sec*Fs/H).astype(int)
C_fix = librosa.util.sync(C, B_fix_frame)

# Visualization
fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.03],
                                          'height_ratios': [1, 2]}, figsize=(8,3.5))

libfmp.b.plot_signal(x, Fs, ax=ax[0,0],
                     title='Signal with fixed-size windowing')
ax[0,1].set_axis_off()
plot_beat_grid(B_fix_sec, ax[0,0])

libfmp.b.plot_matrix(C_fix, ax=[ax[1,0], ax[1,1]], xlabel='Time (frames)',
                     title='Chromgram with fixed-size windowing')
plt.tight_layout()

# %% [markdown]
# We now introduce a function for adaptive windowing, which is informed by note onsets.
# Adaptive windowing leads to clean chroma features that nicely capture the characteristics
# of the eight notes.

# %%
nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=512, H=256,
                                                 gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

peaks, properties = signal.find_peaks(nov, prominence=0.2)
B_adapt_sec = peaks/Fs_nov
B_adapt_frame = (B_adapt_sec*Fs/H).astype(int)
C_adapt = librosa.util.sync(C, B_adapt_frame, aggregate=np.mean)

# Visualization
fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.03],
                                          'height_ratios': [1, 1, 2]}, figsize=(8,5))

libfmp.b.plot_signal(nov, Fs_nov, color='k', ax=ax[0,0],
                    title='Novelty function with estimated boundaries')
ax[0,1].set_axis_off()
plot_beat_grid(B_adapt_sec, ax[0,0])

libfmp.b.plot_signal(x, Fs, ax=ax[1,0],
                    title='Signal with adaptive windowing')
ax[1,1].set_axis_off()
plot_beat_grid(B_adapt_sec, ax[1,0])

libfmp.b.plot_matrix(C_adapt, ax=[ax[2,0], ax[2,1]], xlabel='Time (frames)',
                    title='Chromgram with adaptive windowing')

plt.tight_layout()

# %% [markdown]
# ## Beat-Synchronous Feature Representation
#
# Adaptive windowing techniques based on beat information are of particular importance for
# many music analysis and retrieval applications. In this case, a windowed section is
# determined by two consecutive beat positions, which results in one feature vector per beat.
# Such **beat-synchronous** feature representations have the advantage of possessing a musical
# time axis (given in beats) rather than a physical time axis (given in seconds). This makes
# the feature representation robust to differences in tempo (given in BPM).

# %% [markdown]
# ## Example: Shostakovich

# %%
fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F07_Shostakovich_Waltz-02-Section_IncreasingTempo.wav')

Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512,
                                                 gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

L_nov = len(nov)
N_nov = 500
H_nov = 10
Theta = np.arange(30, 601)
X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs=Fs_nov,
                                                            N=500, H=10, Theta=Theta)
nov_PLP = libfmp.c6.compute_plp(X, Fs_nov, L_nov, N_nov, H_nov, Theta)
peaks, properties = signal.find_peaks(nov_PLP, prominence=0.02)
B_adapt_sec = peaks / Fs_nov

N, H = 1024, 512
X = librosa.stft(y=x, n_fft=N, hop_length=H, win_length=N, center=True, window='hanning')
Y = np.abs(X)
C = librosa.feature.chroma_stft(S=Y, sr=Fs, norm=1, hop_length=H, n_fft=N)
B_fix_frame = (B_adapt_sec*Fs/H).astype(int)
C_adapt = librosa.util.sync(C, B_fix_frame, aggregate=np.mean)

# Visualization
fig, ax = plt.subplots(5, 2, gridspec_kw={'width_ratios': [1, 0.03],
                                          'height_ratios': [1, 2, 1, 1, 2]}, figsize=(10,8))

libfmp.b.plot_signal(x, Fs, ax=ax[0,0], title='Signal')
ax[0,1].set_axis_off()

libfmp.b.plot_matrix(C, ax=[ax[1,0], ax[1,1]], xlabel='Time (frames)',
                     title='Chromgram with fixed-size windowing (frame rate = %0.1f Hz)'%(Fs/H))

libfmp.b.plot_signal(nov_PLP, Fs_nov, color='k', ax=ax[2,0],
                    title='PLP-based novelty function with estimated beat positions')
ax[2,1].set_axis_off()
plot_beat_grid(B_adapt_sec, ax[2,0])

libfmp.b.plot_signal(x, Fs, ax=ax[3,0],
                    title='Signal with beat positions')
ax[3,1].set_axis_off()
plot_beat_grid(B_adapt_sec, ax[3,0])

libfmp.b.plot_matrix(C_adapt, ax=[ax[4,0], ax[4,1]], xlabel='Time (frames)',
                    title='Beat-synchronous chromagram')

plt.tight_layout()

# %% [markdown]
# ## Transient Removal
#
# Next, we show how the knowledge of onset and beat positions can be used in another way for
# improving the quality of audio features. Recall that note onsets often go along with
# **transients** (noise-like energy bursts) spread over the entire spectrum, especially for
# instruments such as the piano, guitar, or percussion. While transients are useful for the
# detection of note onset positions, they may cause undesired artifacts in features that
# capture harmonic or melodic information. To remove these noise-like artifacts while keeping
# the harmonic information, one idea is to exclude a neighborhood around each note onset
# position. To this end, we introduce a parameter $\lambda\in\mathbb{R}$, $0<\lambda<1$, which
# determines the size of the neighborhoods.

# %%
def adaptive_windowing(X, B, neigborhood=1, add_start=False, add_end=False):
    """Apply adaptive windowing [FMP, Section 6.3.3]

    Notebook: C6/C6S3_AdaptiveWindowing.ipynb

    Args:
        X (np.ndarray): Feature sequence
        B (np.ndarray): Beat sequence (spefied in frames)
        neigborhood (float): Parameter specifying relative range considered for windowing (Default value = 1)
        add_start (bool): Add first index of X to beat sequence (if not existent) (Default value = False)
        add_end (bool): Add last index of X to beat sequence (if not existent) (Default value = False)

    Returns:
        X_adapt (np.ndarray): Feature sequence adapted to beat sequence
        B_s (np.ndarray): Sequence specifying start (in frames) of window sections
        B_t (np.ndarray): Sequence specifying end (in frames) of window sections
    """
    len_X = X.shape[1]
    max_B = np.max(B)
    if max_B > len_X:
        print('Beat exceeds length of features sequence (b=%d, |X|=%d)' % (max_B, len_X))
        B = B[B < len_X]
    if add_start:
        if B[0] > 0:
            B = np.insert(B, 0, 0)
    if add_end:
        if B[-1] < len_X:
            B = np.append(B, len_X)
    X_adapt = np.zeros((X.shape[0], len(B)-1))
    B_s = np.zeros(len(B)-1).astype(int)
    B_t = np.zeros(len(B)-1).astype(int)
    for b in range(len(B)-1):
        s = B[b]
        t = B[b+1]
        reduce = np.floor((1 - neigborhood)*(t-s+1)/2).astype(int)
        s = s + reduce
        t = t - reduce
        if s == t:
            t = t + 1
        X_slice = X[:, range(s, t)]
        X_adapt[:, b] = np.mean(X_slice, axis=1)
        B_s[b] = s
        B_t[b] = t
    return X_adapt, B_s, B_t

def compute_plot_adaptive_windowing(x, Fs, H, X, B, neigborhood=1, add_start=False, add_end=False):
    """Compute and plot process for adaptive windowing [FMP, Section 6.3.3]

    Notebook: C6/C6S3_AdaptiveWindowing.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sample Rate
        H (int): Hop size
        X (int): Feature sequence
        B (np.ndarray): Beat sequence (spefied in frames)
        neigborhood (float): Parameter specifying relative range considered for windowing (Default value = 1)
        add_start (bool): Add first index of X to beat sequence (if not existent) (Default value = False)
        add_end (bool): Add last index of X to beat sequence (if not existent) (Default value = False)

    Returns:
        X_adapt (np.ndarray): Feature sequence adapted to beat sequence
    """
    X_adapt, B_s, B_t = adaptive_windowing(X, B, neigborhood=neigborhood,
                                           add_start=add_start, add_end=add_end)

    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.03],
                                              'height_ratios': [1, 3]}, figsize=(10, 4))

    libfmp.b.plot_signal(x, Fs, ax=ax[0, 0], title=r'Adaptive windowing using $\lambda = %0.2f$' % neigborhood)
    ax[0, 1].set_axis_off()
    plot_beat_grid(B_s * H / Fs, ax[0, 0], color='b')
    plot_beat_grid(B_t * H / Fs, ax[0, 0], color='g')
    plot_beat_grid(B * H / Fs, ax[0, 0], color='r')
    for k in range(len(B_s)):
        ax[0, 0].fill_between([B_s[k] * H / Fs, B_t[k] * H / Fs], -1, 1, facecolor='red', alpha=0.1)

    libfmp.b.plot_matrix(X_adapt, ax=[ax[1, 0], ax[1, 1]], xlabel='Time (frames)', ylabel='Frequency (bins)')
    plt.tight_layout()
    return X_adapt

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F24_ScaleMajorC-middle.wav')

Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)
N, H = 1024, 256
X = librosa.stft(y=x, n_fft=N, hop_length=H, win_length=N, center=True,window='hanning')
X = X[0:100,:]
Y = np.log(1+ 10*np.abs(X))

nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=512, H=256,
                                                 gamma=100, M=10, norm=True)
nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

peaks, properties = signal.find_peaks(nov, prominence=0.2)
B_adapt_sec = peaks/Fs_nov

add_start = False
add_end = False
neigborhood = 1
B_adapt_frame = np.round(B_adapt_sec*Fs/H).astype(int)
Y_adapt = compute_plot_adaptive_windowing(x, Fs, H, Y, B_adapt_frame,
                        neigborhood=neigborhood, add_start=add_start, add_end=add_end)
neigborhood = 0.5
Y_adapt = compute_plot_adaptive_windowing(x, Fs, H, Y, B_adapt_frame,
                        neigborhood=neigborhood, add_start=add_start, add_end=add_end)

add_start = True
add_end = True
neigborhood = 0.5
Y_adapt = compute_plot_adaptive_windowing(x, Fs, H, Y, B_adapt_frame,
                        neigborhood=neigborhood, add_start=add_start, add_end=add_end)

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

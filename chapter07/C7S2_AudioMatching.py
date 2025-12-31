# %% [markdown]
# # Audio Matching
#
# Following Section 7.2 of [Müller, FMP, Springer 2015], we discuss in this notebook the task
# of audio matching.

# %% [markdown]
# ## Task Specification and General Requirements
#
# We now address a retrieval task referred to as **audio matching**: given a short query audio
# clip, the goal is to automatically retrieve all excerpts from all recordings within a given
# audio database that **musically** correspond to the query. In this matching scenario, as
# opposed to classic audio identification, one allows semantically motivated variations as
# they typically appear in different performances and arrangements of a piece of music.
#
# For example, two different performances of the same piece may exhibit significant nonlinear
# global and local differences in tempo, articulation, and phrasing. Furthermore, one has to
# deal with considerable spectral variations, which are due to differences in instrumentation,
# dynamics, accentuation, and so on.

# %% [markdown]
# ## Overall Matching Approach
#
# In the audio matching scenario, we are given a query audio fragment $\mathcal{Q}$ and a
# collection of database recordings (represented by a single document $\mathcal{D}$). A
# typical matching approach proceeds along the following lines:
#
# * **First step:** The query $\mathcal{Q}$ and the document $\mathcal{D}$ are converted into
#   sequences of audio features, say $X=(x_1,x_2,\ldots,x_N)$ and $Y=(y_1,y_2,\ldots,y_M)$.
#   The features should capture piece-specific properties (e.g., harmonic and melodic aspects),
#   while being invariant to performance-specific variations.
#
# * **Second step:** Based on the feature sequences $X$ and $Y$, one tries to identify
#   subsequences in $Y$ that are similar to $X$. One may use diagonal matching or subsequence
#   DTW, obtaining a matching curve $\Delta:[0:M-1]\to\mathbb{R}$.
#
# * **Third step:** Using a suitable strategy for selecting local minima of $\Delta$, one
#   derives a ranked list of matching subsequences (called **matches**).

# %%
import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt

sys.path.append('..')
import libfmp.b
import libfmp.c4
import libfmp.c7
%matplotlib inline

def compute_cens_from_file(fn_wav, Fs=22050, N=4410, H=2205, ell=21, d=5):
    """Compute CENS features from file

    Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        fn_wav (str): Filename of wav file
        Fs (scalar): Feature rate of wav file (Default value = 22050)
        N (int): Window size for STFT (Default value = 4410)
        H (int): Hop size for STFT (Default value = 2205)
        ell (int): Smoothing length (Default value = 21)
        d (int): Downsampling factor (Default value = 5)

    Returns:
        X_CENS (np.ndarray): CENS features
        L (int): Length of CENS feature sequence
        Fs_CENS (scalar): Feature rate of CENS features
        x_duration (float): Duration (seconds) of wav file
    """
    x, Fs = librosa.load(fn_wav, sr=Fs)
    x_duration = x.shape[0] / Fs
    X_chroma = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
    X_CENS, Fs_CENS = libfmp.c7.compute_cens_from_chromagram(X_chroma, Fs=Fs/H, ell=ell, d=d)
    L = X_CENS.shape[1]
    return X_CENS, L, Fs_CENS, x_duration

def compute_matching_function_dtw(X, Y, stepsize=2):
    """Compute CENS features from file

    Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        X (np.ndarray): Query feature sequence (given as K x N matrix)
        Y (np.ndarray): Database feature sequence (given as K x M matrix)
        stepsize (int): Parameter for step size condition (1 or 2) (Default value = 2)

    Returns:
        Delta (np.ndarray): DTW-based matching function
        C (np.ndarray): Cost matrix
        D (np.ndarray): Accumulated cost matrix
    """
    C = libfmp.c7.cost_matrix_dot(X, Y)
    if stepsize == 1:
        D = libfmp.c7.compute_accumulated_cost_matrix_subsequence_dtw(C)
    if stepsize == 2:
        D = libfmp.c7.compute_accumulated_cost_matrix_subsequence_dtw_21(C)
    N, M = C.shape
    Delta = D[-1, :] / N
    return Delta, C, D

def matches_dtw(pos, D, stepsize=2):
    """Derives matches from positions for DTW-based strategy

    Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        pos (np.ndarray): End positions of matches
        D (np.ndarray): Accumulated cost matrix
        stepsize (int): Parameter for step size condition (1 or 2) (Default value = 2)

    Returns:
        matches (np.ndarray): Array containing matches (start, end)
    """
    matches = np.zeros((len(pos), 2)).astype(int)
    for k in range(len(pos)):
        t = pos[k]
        matches[k, 1] = t
        if stepsize == 1:
            P = libfmp.c7.compute_optimal_warping_path_subsequence_dtw(D, m=t)
        if stepsize == 2:
            P = libfmp.c7.compute_optimal_warping_path_subsequence_dtw_21(D, m=t)
        s = P[0, 1]
        matches[k, 0] = s
    return matches

def compute_plot_matching_function_DTW(fn_wav_X, fn_wav_Y, fn_ann,
                                       ell=21, d=5, stepsize=2, tau=0.2, num=5, ylim=[0,0.35]):
    ann, _ = libfmp.c4.read_structure_annotation(fn_ann)
    color_ann = {'Theme': [0, 0, 1, 0.1], 'Match': [0, 0, 1, 0.2]}
    X, N, Fs_X, x_duration = compute_cens_from_file(fn_wav_X, ell=ell, d=d)
    Y, M, Fs_Y, y_duration = compute_cens_from_file(fn_wav_Y, ell=ell, d=d)
    Delta, C, D = compute_matching_function_dtw(X, Y, stepsize=stepsize)
    pos = libfmp.c7.mininma_from_matching_function(Delta, rho=2*N//3, tau=tau, num=num)
    matches = matches_dtw(pos, D, stepsize=stepsize)

    fig, ax = plt.subplots(2, 1, gridspec_kw={'width_ratios': [1],
                                              'height_ratios': [1, 1]}, figsize=(8, 4))
    cmap = libfmp.b.compressed_gray_cmap(alpha=-10, reverse=True)
    libfmp.b.plot_matrix(C, Fs=Fs_X, ax=[ax[0]], ylabel='Time (seconds)',
                         title='Cost matrix $C$ with ground truth annotations (blue rectangles)',
                         colorbar=False, cmap=cmap)
    libfmp.b.plot_segments_overlay(ann, ax=ax[0], alpha=0.2, time_max=y_duration,
                                   colors = color_ann, print_labels=False)

    title = r'Matching function $\Delta_\mathrm{DTW}$ with matches (red rectangles)'
    libfmp.b.plot_signal(Delta,  ax=ax[1], Fs=Fs_X, color='k', title=title, ylim=ylim)
    ax[1].grid()
    libfmp.c7.plot_matches(ax[1], matches, Delta, Fs=Fs_X, s_marker='', t_marker='o')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## DTW-Based Audio Matching: Beethoven Example
#
# We apply the matching procedure for our Beethoven example. As a query, we use the Bernstein
# recording of the first theme (first 21 measures). The theme appears once more in the
# repetition of the exposition and, with some musical modifications, in the recapitulation.

# %%
data_dir = os.path.join('..', 'data', 'C7')
fn_wav_all = [os.path.join(data_dir, 'FMP_C7_Audio_Beethoven_Op067-01_Bernstein.wav'),
              os.path.join(data_dir, 'FMP_C7_Audio_Beethoven_Op067-01_Karajan.wav'),
              os.path.join(data_dir, 'FMP_C7_Audio_Beethoven_Op067-01_Scherbakov.wav')]
fn_ann_all = [os.path.join(data_dir, 'FMP_C7_Audio_Beethoven_Op067-01_Bernstein_Theme.csv'),
              os.path.join(data_dir, 'FMP_C7_Audio_Beethoven_Op067-01_Karajan_Theme.csv'),
              os.path.join(data_dir, 'FMP_C7_Audio_Beethoven_Op067-01_Scherbakov_Theme.csv')]
names_all = ['Bernstein', 'Karajan', 'Scherbakov (piano version)']
fn_wav_X = os.path.join(data_dir, 'FMP_C7_Audio_Beethoven_Op067-01_Bernstein_Theme_1.wav')

for f in range(3):
    print('=== Query X: Bernstein (Theme 1); Database Y:', names_all[f],' ===')
    compute_plot_matching_function_DTW(fn_wav_X,  fn_wav_all[f], fn_ann_all[f])

# %% [markdown]
# ## DTW-Based Audio Matching: Shostakovich Example
#
# As a second example, we consider the second Waltz of Shostakovich's Jazz Suite No. 2. This
# piece is of the form $A_1A_2BA_3A_4$, where the $A$-part consists of 38 measures and appears
# four times, each time in a different instrumentation. In part $A_1$ the melody is played by
# saxophone and wood instruments, then in $A_2$ by strings, in $A_3$ by trombone and brass,
# and finally in $A_4$ in a tutti version.

# %%
data_dir = os.path.join('..', 'data', 'C7')
fn_wav_all = [os.path.join(data_dir, 'FMP_C7_Audio_Shostakovich_Waltz-02_Chailly.wav'),
              os.path.join(data_dir, 'FMP_C7_Audio_Shostakovich_Waltz-02_Yablonsky.wav')]
fn_ann_all = [os.path.join(data_dir, 'FMP_C7_Audio_Shostakovich_Waltz-02_Chailly_Theme.csv'),
              os.path.join(data_dir, 'FMP_C7_Audio_Shostakovich_Waltz-02_Yablonsky_Theme.csv')]
names_all = ['Chailly', 'Yablonsky']
fn_wav_X = os.path.join(data_dir, 'FMP_C7_Audio_Shostakovich_Waltz-02_Chailly_Theme_1.wav')

for f in range(2):
    print('=== Query X: Chailly (A1, 16 measures); Database Y:', names_all[f],' ===')
    compute_plot_matching_function_DTW(fn_wav_X,  fn_wav_all[f], fn_ann_all[f], ylim=[0, 0.25])

# %% [markdown]
# Looking at the results:
#
# * All four occurrences in both of the versions appear as the top four matches.
# * Even though the Yablonsky version is faster than the Chailly version, these tempo
#   variations are successfully handled by the DTW-based matching strategy.
# * In both versions, the occurrence in $A_3$ (trombone version) has the largest
#   $\Delta_\mathrm{DTW}$-distance. This is due to the fact that the spectra of low-pitched
#   instruments generally exhibit phenomena such as oscillations and smearing.

# %%
fn_wav_X = os.path.join(data_dir, 'FMP_C7_Audio_Shostakovich_Waltz-02_Chailly_Theme_3.wav')

for f in range(2):
    print('=== Query X: Chailly (A3, 16 measures); Database Y:',names_all[f],' ===')
    compute_plot_matching_function_DTW(fn_wav_X,  fn_wav_all[f], fn_ann_all[f], ylim=[0,0.25])

# %% [markdown]
# The quality of the matching results also crucially depends on the length of the query.

# %%
fn_wav_X = os.path.join(data_dir, 'FMP_C7_Audio_Shostakovich_Waltz-02_Chailly_Theme_3_32.wav')
fn_ann_all = [os.path.join(data_dir, 'FMP_C7_Audio_Shostakovich_Waltz-02_Chailly_Theme_32.csv'),
              os.path.join(data_dir, 'FMP_C7_Audio_Shostakovich_Waltz-02_Yablonsky_Theme_32.csv')]


for f in range(2):
    print('=== Query X: Chailly (A3, 32 measures); Database Y:',names_all[f],' ===')
    compute_plot_matching_function_DTW(fn_wav_X,  fn_wav_all[f], fn_ann_all[f], ylim=[0, 0.2])

# %% [markdown]
# ## Transposition-Invariant Matching Function
#
# In retrieval applications, one may want to identify audio excerpts even if they are played
# in a different musical key. As an example, let us consider the song "In the Year 2525" by
# Zager and Evans. The song has the overall musical structure $IV_1V_2V_3V_4V_5V_6V_7BV_8O$.
# While the first four verse sections are in the same musical key, $V_5$ and $V_6$ are
# transposed by one semitone upwards, and $V_7$ and $V_8$ are transposed by two semitones
# upwards.
#
# Using the **cyclic shift operator** $\rho:\mathbb{R}^{12} \to \mathbb{R}^{12}$, one can
# simulate transpositions. The **transposition-invariant matching function** $\Delta^\mathrm{TI}$
# is obtained by setting
#
# $$\Delta^\mathrm{TI}(m):=  \min_{i\in [0:11]} \Delta^{i}(m)$$

# %%
def compute_matching_function_dtw_ti(X, Y, cyc=np.arange(12), stepsize=2):
    """Compute transposition-invariant matching function

    Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        X (np.ndarray): Query feature sequence (given as K x N matrix)
        Y (np.ndarray): Database feature sequence (given as K x M matrix)
        cyc (np.nda(rray): Set of cyclic shift indices to be considered (Default value = np.arange(12))
        stepsize (int): Parameter for step size condition (1 or 2) (Default value = 2)

    Returns:
        Delta_TI (np.ndarray): Transposition-invariant matching function
        Delta_ind (np.ndarray): Cost-minimizing indices
        Delta_cyc (np.ndarray): Array containing all matching functions
    """
    M = Y.shape[1]
    num_cyc = len(cyc)
    Delta_cyc = np.zeros((num_cyc, M))
    for k in range(num_cyc):
        X_cyc = np.roll(X, k, axis=0)
        Delta_cyc[k, :], C, D = compute_matching_function_dtw(X_cyc, Y, stepsize=stepsize)
    Delta_TI = np.min(Delta_cyc, axis=0)
    Delta_ind = np.argmin(Delta_cyc, axis=0)
    return Delta_TI, Delta_ind, Delta_cyc

data_dir = os.path.join('..', 'data', 'C7')
fn_wav = os.path.join(data_dir, 'FMP_C7_Audio_ZagerEvans_InTheYear2525.wav')
fn_ann = os.path.join(data_dir, 'FMP_C7_Audio_ZagerEvans_InTheYear2525.csv')
fn_wav_X = os.path.join(data_dir, 'FMP_C7_Audio_ZagerEvans_InTheYear2525_Part-V1.wav')

ann, _ = libfmp.c4.read_structure_annotation(fn_ann)
ann_color = {'I': 'white', 'V1': 'red', 'V2': 'red', 'V3': 'red', 'V4': 'red', 'V5': 'green', 'V6': 'green',
             'V7': 'blue', 'B': 'white', 'V8': 'blue', 'O': 'gray', '': 'white'}

X, N, Fs_X, x_duration = compute_cens_from_file(fn_wav_X, ell=21, d=5)
Y, M, Fs_Y, y_duration = compute_cens_from_file(fn_wav, ell=21, d=5)

Delta_TI, Delta_ind, Delta_cyc = compute_matching_function_dtw_ti(X, Y)
pos = libfmp.c7.mininma_from_matching_function(Delta_TI, rho=2*N//3, tau=0.1, num=8)

fig, ax = plt.subplots(6, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 0.25]})

color_set = ['red', 'green', 'blue', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray']
for k in range(4):
    libfmp.b.plot_signal(Delta_cyc[k,:], ax=ax[k], xlabel='', ylabel = r'$\Delta^{%d}$' % k,
                         color=color_set[k], ylim=[0, 0.3])
    ax[k].grid()

for k in range(12):
    libfmp.b.plot_signal(Delta_cyc[k,:], ax=ax[4], color=color_set[k], ylim=[0, 0.3])

libfmp.b.plot_signal(Delta_TI, ax=ax[4], color='k', linewidth='3', ylim=[0, 0.3],
                     ylabel = r'$\Delta^{\mathrm{TI}}$', xlabel='')
ax[4].grid()
libfmp.b.plot_segments(ann, ax=ax[5], nontime_axis=False, adjust_nontime_axislim=False,
                       colors=ann_color, alpha=0.25)
ax[4].plot(pos, Delta_TI[pos], 'ro')
ax[5].set_xlabel('Time (seconds)')
plt.tight_layout()

# %% [markdown]
# ## Further Notes
#
# There are many design choices when implementing an audio matching procedure:
#
# * Using chroma-based features, we presented a system suitable for identifying harmonically
#   similar sections of Western music.
#
# * The quality of the matching results depends on the length of the query. For many Western
#   classical music pieces, a query length of at least 20 seconds yields reasonable results.
#
# * Subsequence DTW introduces flexibility to compensate for temporal deformations. When
#   using $\Sigma=\{(2, 1), (1, 2), (1, 1)\}$, it turns out to constitute a good compromise
#   between temporal flexibility and robustness.
#
# * Diagonal matching with multiple queries may have advantages when accelerating the
#   matching process using indexing techniques.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Frank Zalkow.

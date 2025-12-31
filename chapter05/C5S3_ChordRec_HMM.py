# %% [markdown]
# # HMM-Based Chord Recognition
#
# Following Section 5.3.4 of [Müller, FMP, Springer 2015], we discuss in this notebook an HMM-based
# approach for chord recognition. The idea of using HMMs for chord recognition was originally
# introduced by Sheh and Ellis.

# %% [markdown]
# ## Introduction
#
# We now show how the concept of HMMs can be applied to improve automated chord recognition.
# First of all, we need to create an HMM that suitably models our chord recognition problem.
# Generally, an HMM is specified by the parameters $\Theta:=(\mathcal{A},A,C,\mathcal{B},B)$.
# In the chord recognition context, the set of states is used to model the various chord types
# that are allowed in the recognition problem. We consider only the twelve major and twelve
# minor triads, thus setting:
#
# $$\mathcal{A} = \{\mathbf{C},\mathbf{C}^\sharp,\ldots,\mathbf{B},\mathbf{Cm},\mathbf{Cm^\sharp},\ldots,\mathbf{Bm}\}$$
#
# In this case, the HMM consists of $I=24$ states. In the remainder of this notebook, we do the following:
#
# * First, we explain how to explicitly create an HMM by specifying the other HMM parameters in a
#   musically informed fashion.
# * Second, we apply this HMM for chord recognition. The input (observation sequence) of the HMM is a
#   chromagram representation of the music recording. Applying the Viterbi Algorithm, we derive an
#   optimal state sequence (consisting of chord labels) that best explains the chroma sequence.

# %% [markdown]
# ## Specification of Emission Likelihoods
#
# In our chord recognition scenario, the observations are chroma vectors that have previously been
# extracted from the given audio recording. We use an HMM variant where we replace the discrete
# output space by the continuous feature space $\mathcal{F}=\mathbb{R}^{12}$ and the emission
# probability matrix $B$ by likelihood functions.
#
# Let $s:\mathcal{F} \times \mathcal{F} \to [0,1]$ be the similarity measure defined by the
# inner product of normalized chroma vectors:
#
# $$s(x, y) = \frac{\langle x,y\rangle}{\|x\|_2\cdot\|y\|_2}$$
#
# Based on $I=24$ major and minor triads, we consider the binary chord templates $\mathbf{t}_i\in \mathcal{F}$
# for $i\in [1:I]$. Then, we define the state-dependent likelihood function by:
#
# $$b_i(x) := \frac{s(x, \mathbf{t}_i)}{\sum_{j\in[1:I]}s(x, \mathbf{t}_j)}$$

# %%
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.linalg import circulant
from numba import jit

import sys
sys.path.append('..')
import libfmp.b
from libfmp.c5 import get_chord_labels
%matplotlib inline


# Specify
fn_wav = os.path.join('..', 'data', 'C5', 'FMP_C5_F20_Bach_BWV846-mm1-4_Fischer.wav')
fn_ann = os.path.join('..', 'data', 'C5', 'FMP_C5_F20_Bach_BWV846-mm1-4_Fischer_ChordAnnotations.csv')
color_ann = {'C': [1, 0.5, 0, 1], 'G': [0, 1, 0, 1], 'Dm': [1, 0, 0, 1], 'N': [1, 1, 1, 1]}

N = 4096
H = 1024
X, Fs_X, x, Fs, x_dur = \
    libfmp.c5.compute_chromagram_from_filename(fn_wav, N=N, H=H, gamma=0.1, version='STFT')
N_X = X.shape[1]

# Chord recogntion
chord_sim, chord_max = libfmp.c5.chord_recognition_template(X, norm_sim='1')
chord_labels = libfmp.c5.get_chord_labels(nonchord=False)

# Annotations
chord_labels = libfmp.c5.get_chord_labels(ext_minor='m', nonchord=False)
ann_matrix, ann_frame, ann_seg_frame, ann_seg_ind, ann_seg_sec = \
    libfmp.c5.convert_chord_ann_matrix(fn_ann, chord_labels, Fs=Fs_X, N=N_X, last=True)
#P, R, F, TP, FP, FN = libfmp.c5.compute_eval_measures(ann_matrix, chord_max)

# Plot
cmap = libfmp.b.compressed_gray_cmap(alpha=1, reverse=False)
fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.03],
                                          'height_ratios': [1.5, 3, 0.2]}, figsize=(9, 7))

libfmp.b.plot_chromagram(X, ax=[ax[0, 0], ax[0, 1]], Fs=Fs_X, clim=[0, 1], xlabel='',
                         title='Observation sequence (chromagram with feature rate = %0.1f Hz)' % (Fs_X))
libfmp.b.plot_segments_overlay(ann_seg_sec, ax=ax[0, 0], time_max=x_dur,
                               print_labels=False, colors=color_ann, alpha=0.1)

libfmp.b.plot_matrix(chord_sim, ax=[ax[1, 0], ax[1, 1]], Fs=Fs_X, clim=[0, np.max(chord_sim)],
                     title='Likelihood matrix (time–chord representation)',
                     ylabel='Chord', xlabel='')
ax[1, 0].set_yticks(np.arange(len(chord_labels)))
ax[1, 0].set_yticklabels(chord_labels)
libfmp.b.plot_segments_overlay(ann_seg_sec, ax=ax[1, 0], time_max=x_dur,
                               print_labels=False, colors=color_ann, alpha=0.1)

libfmp.b.plot_segments(ann_seg_sec, ax=ax[2, 0], time_max=x_dur, time_label='Time (seconds)',
                       colors=color_ann,  alpha=0.3)
ax[2,1].axis('off')
plt.tight_layout()

# %% [markdown]
# ## Specification of Transition Probabilities
#
# In music, certain chord transitions are more likely than others. This observation is our main
# motivation to employ HMMs, where the first-order temporal relationships between the various
# chords can be captured by the transition probability matrix $A$. The coefficients $a_{i,i}$
# express the probability of staying in state $\alpha_{i}$ (self-transition probabilities).
#
# A transition probability matrix can be specified in many ways. For example:
# * The matrix may be defined manually by a music expert based on rules from harmony theory
# * It may be generated automatically by estimating the transition probabilities from labeled data
#
# We show three different transition matrices:
# * One learned from labeled training data based on the Beatles collection
# * A transposition-invariant transition probability matrix
# * A uniform transition probability matrix with a large value on the main diagonal

# %%
def plot_transition_matrix(A, log=True, ax=None, figsize=(6, 5), title='',
                           xlabel='State (chord label)', ylabel='State (chord label)',
                           cmap='gray_r', quadrant=False):
    """Plot a transition matrix for 24 chord models (12 major and 12 minor triads)

    Notebook: C5/C5S3_ChordRec_HMM.ipynb

    Args:
        A: Transition matrix
        log: Show log probabilities (Default value = True)
        ax: Axis (Default value = None)
        figsize: Width, height in inches (only used when ax=None) (Default value = (6, 5))
        title: Title for plot (Default value = '')
        xlabel: Label for x-axis (Default value = 'State (chord label)')
        ylabel: Label for y-axis (Default value = 'State (chord label)')
        cmap: Color map (Default value = 'gray_r')
        quadrant: Plots additional lines for C-major and C-minor quadrants (Default value = False)

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
        im: The image plot
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax = [ax]

    if log is True:
        A_plot = np.log(A)
        cbar_label = 'Log probability'
        clim = [-6, 0]
    else:
        A_plot = A
        cbar_label = 'Probability'
        clim = [0, 1]
    im = ax[0].imshow(A_plot, origin='lower', aspect='equal', cmap=cmap, interpolation='nearest')
    im.set_clim(clim)
    plt.sca(ax[0])
    cbar = plt.colorbar(im)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(title)
    cbar.ax.set_ylabel(cbar_label)

    chord_labels = get_chord_labels()
    chord_labels_squeezed = chord_labels.copy()
    for k in [1, 3, 6, 8, 10, 11, 13, 15, 17, 18, 20, 22]:
        chord_labels_squeezed[k] = ''

    ax[0].set_xticks(np.arange(24))
    ax[0].set_yticks(np.arange(24))
    ax[0].set_xticklabels(chord_labels_squeezed)
    ax[0].set_yticklabels(chord_labels)

    if quadrant is True:
        ax[0].axvline(x=11.5, ymin=0, ymax=24, linewidth=2, color='r')
        ax[0].axhline(y=11.5, xmin=0, xmax=24, linewidth=2, color='r')

    return fig, ax, im

# Load transition matrix estimated on the basis of the Beatles collection
fn_csv = os.path.join('..', 'data', 'C5', 'FMP_C5_transitionMatrix_Beatles.csv')
A_est_df = pd.read_csv(fn_csv, delimiter=';')
A_est = A_est_df.to_numpy('float64')

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1],
                                          'height_ratios': [1]},
                       figsize=(10, 3.8))

plot_transition_matrix(A_est, log=False, ax=[ax[0]], title='Transition matrix')
plot_transition_matrix(A_est, ax=[ax[1]], title='Transition matrix with log probabilities')
plt.tight_layout()

# %% [markdown]
# ## Transposition-Invariant Transition Matrix
#
# To obtain the transposition-invariant transition matrix, we simulate the cyclic chroma shifts
# on the matrix-level by cyclically shifting and averaging the four quadrants (defined by the
# major-chord and minor-chord regions) of the original matrix.

# %%
def matrix_circular_mean(A):
    """Computes circulant matrix with mean diagonal sums

    Notebook: C5/C5S3_ChordRec_HMM.ipynb

    Args:
        A (np.ndarray): Square matrix

    Returns:
        A_mean (np.ndarray): Circulant output matrix
    """
    N = A.shape[0]
    A_shear = np.zeros((N, N))
    for n in range(N):
        A_shear[:, n] = np.roll(A[:, n], -n)
    circ_sum = np.sum(A_shear, axis=1)
    A_mean = circulant(circ_sum) / N
    return A_mean

def matrix_chord24_trans_inv(A):
    """Computes transposition-invariant matrix for transition matrix
    based 12 major chords and 12 minor chords

    Notebook: C5/C5S3_ChordRec_HMM.ipynb

    Args:
        A (np.ndarray): Input transition matrix

    Returns:
        A_ti (np.ndarray): Output transition matrix
    """
    A_ti = np.zeros(A.shape)
    A_ti[0:12, 0:12] = matrix_circular_mean(A[0:12, 0:12])
    A_ti[0:12, 12:24] = matrix_circular_mean(A[0:12, 12:24])
    A_ti[12:24, 0:12] = matrix_circular_mean(A[12:24, 0:12])
    A_ti[12:24, 12:24] = matrix_circular_mean(A[12:24, 12:24])
    return A_ti


A_ti = matrix_chord24_trans_inv(A_est)

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1],
                                          'height_ratios': [1]},
                       figsize=(10, 3.8))

plot_transition_matrix(A_est, ax=[ax[0]], quadrant=True,
                       title='Transition matrix')
plot_transition_matrix(A_ti, ax=[ax[1]], quadrant=True,
                       title='Transposition-invariant transition matrix')
plt.tight_layout()

# %% [markdown]
# ## Uniform Transition Matrix
#
# We provide a function for generating a uniform transition probability matrix. This function
# has a parameter $p\in[0,1]$ that determines the probability for self transitions (the value
# on the main diagonal). The probabilities on the remaining positions are set such that the
# resulting matrix is a probability matrix (i.e., all rows and columns sum to one).

# %%
def uniform_transition_matrix(p=0.01, N=24):
    """Computes uniform transition matrix

    Notebook: C5/C5S3_ChordRec_HMM.ipynb

    Args:
        p (float): Self transition probability (Default value = 0.01)
        N (int): Column and row dimension (Default value = 24)

    Returns:
        A (np.ndarray): Output transition matrix
    """
    off_diag_entries = (1-p) / (N-1)     # rows should sum up to 1
    A = off_diag_entries * np.ones([N, N])
    np.fill_diagonal(A, p)
    return A

fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1],
                                          'height_ratios': [1]},
                       figsize=(10, 3.8))

p = 0.5
A_uni = uniform_transition_matrix(p)
plot_transition_matrix(A_uni, ax=[ax[0]], title='Uniform transition matrix (p=%0.2f)' % p)
p = 0.9
A_uni = uniform_transition_matrix(p)
plot_transition_matrix(A_uni, ax=[ax[1]], title='Uniform transition matrix (p=%0.2f)' % p)
plt.tight_layout()

# %% [markdown]
# ## HMM-Based Chord Recognition
#
# We now present an experiment that demonstrates the effect of applying HMMs to our chord
# recognition scenario. We use the following setting:
#
# * As observation sequence $O$, we use a sequence of chroma vectors.
# * As for the transition probability matrix $A$, we use a uniform transition matrix.
# * As for the initial state probability vector $C$, we use a uniform distribution.
# * As for the emission probability matrix $B$, we replace them by the likelihood matrix $B[O]$.
# * The frame-wise chord recognition results is given by the state sequence computed by the
#   Viterbi algorithm.

# %%
@jit(nopython=True)
def viterbi_log_likelihood(A, C, B_O):
    """Viterbi algorithm (log variant) for solving the uncovering problem

    Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        A (np.ndarray): State transition probability matrix of dimension I x I
        C (np.ndarray): Initial state distribution  of dimension I
        B_O (np.ndarray): Likelihood matrix of dimension I x N

    Returns:
        S_opt (np.ndarray): Optimal state sequence of length N
        S_mat (np.ndarray): Binary matrix representation of optimal state sequence
        D_log (np.ndarray): Accumulated log probability matrix
        E (np.ndarray): Backtracking matrix
    """
    I = A.shape[0]    # Number of states
    N = B_O.shape[1]  # Length of observation sequence
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_O_log = np.log(B_O + tiny)

    # Initialize D and E matrices
    D_log = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D_log[:, 0] = C_log + B_O_log[:, 0]

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + D_log[:, n-1]
            D_log[i, n] = np.max(temp_sum) + B_O_log[i, n]
            E[i, n-1] = np.argmax(temp_sum)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N-2, -1, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    # Matrix representation of result
    S_mat = np.zeros((I, N)).astype(np.int32)
    for n in range(N):
        S_mat[S_opt[n], n] = 1

    return S_mat, S_opt, D_log, E

A = uniform_transition_matrix(p=0.5)
C = 1 / 24 * np.ones((1, 24))
B_O = chord_sim
chord_HMM, _, _, _ = viterbi_log_likelihood(A, C, B_O)

P, R, F, TP, FP, FN = libfmp.c5.compute_eval_measures(ann_matrix, chord_HMM)
title = 'HMM-Based approach (N=%d, TP=%d, FP=%d, FN=%d, P=%.2f, R=%.2f, F=%.2f)' % (N_X, TP, FP, FN, P, R, F)
fig, ax, im = libfmp.c5.plot_matrix_chord_eval(ann_matrix, chord_HMM, Fs=1,
                     title=title, ylabel='Chord', xlabel='Time (frames)', chord_labels=chord_labels)
plt.tight_layout()
plt.show()

P, R, F, TP, FP, FN = libfmp.c5.compute_eval_measures(ann_matrix, chord_max)
title = 'Template-based approach (N=%d, TP=%d, FP=%d, FN=%d, P=%.2f, R=%.2f, F=%.2f)' %\
    (N_X, TP, FP, FN, P, R, F)
fig, ax, im = libfmp.c5.plot_matrix_chord_eval(ann_matrix, chord_max, Fs=1,
                     title=title, ylabel='Chord', xlabel='Time (frames)', chord_labels=chord_labels)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Observations
#
# In this example, the HMM-based chord recognizer clearly outperforms the template-based approach.
# The improvements in the HMM-based approach come specifically from the transition model that
# introduces context-sensitive smoothing. In the case of high self-transition probabilities,
# a chord recognizer tends to stay in the current chord rather than change to another one,
# which can be regarded as a kind of smoothing.

# %% [markdown]
# ## Prefiltering vs. Postfiltering
#
# Applying longer window sizes amounts to temporal smoothing of the observation sequence.
# Since this smoothing is performed prior to the pattern matching step, we call this strategy
# prefiltering. As opposed to prefiltering, the HMM-based approach leaves the feature
# representation untouched. Furthermore, the smoothing is performed in combination with the
# pattern matching step. For this reason, we also call this approach postfiltering.

# %% [markdown]
# ## Further Notes
#
# In this notebook, we introduced a basic HMM-based approach for chord recognition. In our
# simplistic model, we used 24 states that correspond to the 12 major and 12 minor triads
# and fixed the HMM parameters explicitly using musical knowledge. In general, there is a
# delicate interplay of the various feature extraction, filtering, and pattern matching
# components composing a chord recognition system.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Christof Weiß.

# %% [markdown]
# # Audio Thumbnailing
#
# Following Section 4.3 of [Müller, FMP, Springer 2015], we discuss in this notebook a
# music processing task referred to as audio thumbnailing.

# %% [markdown]
# ## Introduction
#
# A prominent subproblem of music structure analysis is known as **audio thumbnailing**.
# Given a music recording, the objective is to automatically determine the most
# representative section, which may serve as a kind of "preview" giving a listener a
# first impression of the song or piece of music. We describe a **repetition-based approach**.

# %% [markdown]
# ## Requirements on SSM
#
# The idea of our fitness measure is to simultaneously establish all relations between
# a given segment and its approximate repetitions. To this end, a self-similarity matrix
# (SSM) is required.

# %%
import numpy as np
import os, sys, librosa, math
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import IPython.display as ipd
import pandas as pd
from numba import jit
from matplotlib.colors import ListedColormap
sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c3
import libfmp.c4

%matplotlib inline

def colormap_penalty(penalty=-2, cmap=libfmp.b.compressed_gray_cmap(alpha=5)):
    """Extend colormap with white color between the penalty value and zero

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        penalty (float): Negative number (Default value = -2.0)
        cmap (mpl.colors.Colormap): Original colormap (Default value = libfmp.b.compressed_gray_cmap(alpha=5))

    Returns:
        cmap_penalty (mpl.colors.Colormap): Extended colormap
    """
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap, 128)
    cmap_matrix = cmap(np.linspace(0, 1, 128))[:, :3]
    num_row = int(np.abs(penalty)*128)
    cmap_penalty = np.concatenate((np.ones((num_row, 3)), cmap_matrix), axis=0)
    cmap_penalty = ListedColormap(cmap_penalty)

    return cmap_penalty

def normalization_properties_ssm(S):
    """Normalizes self-similartiy matrix to fulfill S(n,n)=1.
    Yields a warning if max(S)<=1 is not fulfilled

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        S (np.ndarray): Self-similarity matrix (SSM)

    Returns:
        S_normalized (np.ndarray): Normalized self-similarity matrix
    """
    S_normalized = S.copy()
    N = S_normalized.shape[0]
    for n in range(N):
        S_normalized[n, n] = 1
        max_S = np.max(S_normalized)
    if max_S > 1:
        print('Normalization condition for SSM not fulfill (max > 1)')
    return S_normalized

def plot_ssm_ann(S, ann, Fs=1, cmap='gray_r', color_ann=[], ann_x=True, ann_y=True,
                 fontsize=12, figsize=(5, 4.5), xlabel='', ylabel='', title=''):
    """Plot SSM and annotations (horizontal and vertical as overlay)

    Notebook: C4/C4S3_AudioThumbnailing.ipynb
    """
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                              'height_ratios': [1, 0.1]}, figsize=figsize)

    fig_im, ax_im, im = libfmp.b.plot_matrix(S, Fs=Fs, Fs_F=Fs,
                                             ax=[ax[0, 0], ax[0, 1]], cmap=cmap,
                                             xlabel='', ylabel='', title='')
    ax[0, 0].set_ylabel(ylabel)
    ax[0, 0].set_xlabel(xlabel)
    ax[0, 0].set_title(title)
    if ann_y:
        libfmp.b.plot_segments_overlay(ann, ax=ax_im[0], direction='vertical',
                                       time_max=S.shape[0]/Fs, print_labels=False,
                                       colors=color_ann, alpha=0.05)
    if ann_x:
        libfmp.b.plot_segments(ann, ax=ax[1, 0], time_max=S.shape[0]/Fs, colors=color_ann,
                               time_axis=False, fontsize=fontsize)
    else:
        ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    plt.tight_layout()
    return fig, ax, im


fn_wav = os.path.join('..', 'data', 'C4', 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.wav')
tempo_rel_set = libfmp.c4.compute_tempo_rel_set(0.66, 1.5, 5)
penalty = -2
x, x_duration, X, Fs_feature, S, I = libfmp.c4.compute_sm_from_filename(fn_wav, L=21, H=5,
                        L_smooth=12, tempo_rel_set=tempo_rel_set, penalty=penalty, thresh= 0.15)
S = normalization_properties_ssm(S)

fn_ann_color = 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.csv'
fn_ann = os.path.join('..', 'data', 'C4', fn_ann_color)
ann_frames, color_ann = libfmp.c4.read_structure_annotation(fn_ann, fn_ann_color, Fs=Fs_feature)

cmap_penalty = colormap_penalty(penalty=penalty)
fig, ax, im = plot_ssm_ann(S, ann_frames, Fs=1, color_ann=color_ann, cmap=cmap_penalty,
                       xlabel='Time (frames)', ylabel='Time (frames)')

# %% [markdown]
# ## Path Family
#
# We start by extending some notions as defined for SSMs. A **path family** over a
# segment alpha is a set of paths over alpha, where the induced segments are pairwise
# disjoint.

# %%
def plot_path_family(ax, path_family, Fs=1, x_offset=0, y_offset=0, proj_x=True, w_x=7, proj_y=True, w_y=7):
    """Plot path family into a given axis

    Notebook: C4/C4S3_AudioThumbnailing.ipynb
    """
    for path in path_family:
        y = [(path[i][0] + y_offset)/Fs for i in range(len(path))]
        x = [(path[i][1] + x_offset)/Fs for i in range(len(path))]
        ax.plot(x, y, "o", color=[0, 0, 0], linewidth=3, markersize=5)
        ax.plot(x, y, '.', color=[0.7, 1, 1], linewidth=2, markersize=6)
    if proj_y:
        for path in path_family:
            y1 = path[0][0]/Fs
            y2 = path[-1][0]/Fs
            ax.add_patch(plt.Rectangle((0, y1), w_y, y2-y1, linewidth=1,
                                       facecolor=[0, 1, 0], edgecolor=[0, 0, 0]))
    if proj_x:
        for path in path_family:
            x1 = (path[0][1] + x_offset)/Fs
            x2 = (path[-1][1] + x_offset)/Fs
            ax.add_patch(plt.Rectangle((x1, 0), x2-x1, w_x, linewidth=1,
                                       facecolor=[0, 0, 1], edgecolor=[0, 0, 0]))

# Manually defined path family
seg_sec = [20, 80]
seg = [int(seg_sec[0]*Fs_feature), int(seg_sec[1]*Fs_feature)]
path_1 = [np.array([i+seg[0],i]) for i in range(0, seg[-1]-seg[0]+1)]
path_2 = [np.array([int(i+130*Fs_feature),i]) for i in range(0, seg[-1]-seg[0]+1)]
path_family = [path_1, path_2]
print('Segment:', seg)
fig, ax, im = plot_ssm_ann(S, ann_frames, Fs=1, color_ann=color_ann, cmap=cmap_penalty,
                       xlabel='Time (frames)', ylabel='Time (frames)')
plot_path_family(ax[0,0], path_family, Fs=1, x_offset=seg[0])

# %% [markdown]
# ## Coverage
#
# The **coverage** of a segment family is the sum of the lengths of its segments.

# %%
def compute_induced_segment_family_coverage(path_family):
    """Compute induced segment family and coverage from path family

    Notebook: C4/C4S3_AudioThumbnailing.ipynb
    """
    num_path = len(path_family)
    coverage = 0
    if num_path > 0:
        segment_family = np.zeros((num_path, 2), dtype=int)
        for n in range(num_path):
            segment_family[n, 0] = path_family[n][0][0]
            segment_family[n, 1] = path_family[n][-1][0]
            coverage = coverage + segment_family[n, 1] - segment_family[n, 0] + 1
    else:
        segment_family = np.empty

    return segment_family, coverage

segment_family, coverage = compute_induced_segment_family_coverage(path_family)
print('Segment (alpha):', seg)
print('Induced segment family:')
print(segment_family)
print('Coverage: %d'%coverage)

# %% [markdown]
# ## Score and Optimality
#
# The **score** of a path family is the sum of the scores of its paths.

# %% [markdown]
# ## Computation of Optimal Path Families
#
# There is an efficient algorithm using dynamic programming to compute an optimal
# path family.

# %%
@jit(nopython=True)
def compute_accumulated_score_matrix(S_seg):
    """Compute the accumulated score matrix

    Notebook: C4/C4S3_AudioThumbnailing.ipynb
    """
    inf = math.inf
    N = S_seg.shape[0]
    M = S_seg.shape[1]+1

    # Initializing score matrix
    D = -inf * np.ones((N, M), dtype=np.float64)
    D[0, 0] = 0.
    D[0, 1] = D[0, 0] + S_seg[0, 0]

    # Dynamic programming
    for n in range(1, N):
        D[n, 0] = max(D[n-1, 0], D[n-1, -1])
        D[n, 1] = D[n, 0] + S_seg[n, 0]
        for m in range(2, M):
            D[n, m] = S_seg[n, m-1] + max(D[n-1, m-1], D[n-1, m-2], D[n-2, m-1])

    # Score of optimal path family
    score = np.maximum(D[N-1, 0], D[N-1, M-1])

    return D, score

# %%
@jit(nopython=True)
def compute_optimal_path_family(D):
    """Compute an optimal path family given an accumulated score matrix

    Notebook: C4/C4S3_AudioThumbnailing.ipynb
    """
    # Initialization
    inf = math.inf
    N = int(D.shape[0])
    M = int(D.shape[1])

    path_family = []
    path = []

    n = N - 1
    if(D[n, M-1] < D[n, 0]):
        m = 0
    else:
        m = M-1
        path_point = (N-1, M-2)
        path.append(path_point)

    # Backtracking
    while n > 0 or m > 0:
        # obtaining the set of possible predecessors given our current position
        if(n <= 2 and m <= 2):
            predecessors = [(n-1, m-1)]
        elif(n <= 2 and m > 2):
            predecessors = [(n-1, m-1), (n-1, m-2)]
        elif(n > 2 and m <= 2):
            predecessors = [(n-1, m-1), (n-2, m-1)]
        else:
            predecessors = [(n-1, m-1), (n-2, m-1), (n-1, m-2)]

        # case for the first row
        if n == 0:
            cell = (0, m-1)
        # case for the elevator column
        elif m == 0:
            if D[n-1, M-1] > D[n-1, 0]:
                cell = (n-1, M-1)
                path_point = (n-1, M-2)
                if(len(path) > 0):
                    path.reverse()
                    path_family.append(path)
                path = [path_point]
            else:
                cell = (n-1, 0)
        # case for m=1
        elif m == 1:
            cell = (n, 0)
        # regular case
        else:
            max_val = -inf
            for i, cur_predecessor in enumerate(predecessors):
                if(max_val < D[cur_predecessor[0], cur_predecessor[1]]):
                    max_val = D[cur_predecessor[0], cur_predecessor[1]]
                    cell = cur_predecessor

            path_point = (cell[0], cell[1]-1)
            path.append(path_point)

        (n, m) = cell

    # adding last path to the path family
    path.reverse()
    path_family.append(path)
    path_family.reverse()

    return path_family

# %%
def plot_matrix_seg(ax, M, seg, title='', xlabel='', cmap='gray_r'):
    libfmp.b.plot_matrix(M, Fs=1, ax=[ax], cmap=cmap,
        xlabel=xlabel, ylabel='Time (frames)', title=title)
    ax.set_xticks([0, M.shape[1]-1])

seg_sec = [41,68]
seg = [int(seg_sec[0]*Fs_feature), int(seg_sec[1]*Fs_feature)]
print('Segment:', seg)
print('Length of segment: M = ', seg[1]-seg[0]+1)
S_seg = S[:,seg[0]:seg[1]+1]
D, score = compute_accumulated_score_matrix(S_seg)
path_family = compute_optimal_path_family(D)

plt.figure(figsize=(8, 8))
ax = plt.subplot(2, 2, 1)
title = r'$\mathcal{S}^{\alpha}$'
plot_matrix_seg(ax, S_seg, seg, title=title, xlabel='', cmap=cmap_penalty)

ax = plt.subplot(2, 2, 2)
title = r'$D$'
plot_matrix_seg(ax, D, seg, title=title)

ax = plt.subplot(2, 2, 3)
title = r'$\mathcal{S}^{\alpha}$ with optimal path family'
plot_matrix_seg(ax, S_seg, seg, title=title, xlabel='', cmap=cmap_penalty)
plot_path_family(ax, path_family, x_offset=0, w_y=1)

ax = plt.subplot(2, 2, 4)
title = r'$D$ with optimal path family'
plot_matrix_seg(ax, D, seg, title=title)
plot_path_family(ax, path_family, x_offset=1, w_y=1)

plt.tight_layout()

# %% [markdown]
# ## Fitness Measure
#
# The **fitness** of a segment is the harmonic mean between the normalized score
# and normalized coverage.

# %%
def compute_fitness(path_family, score, N):
    """Compute fitness measure and other metrics from path family

    Notebook: C4/C4S3_AudioThumbnailing.ipynb
    """
    eps = 1e-16
    num_path = len(path_family)
    M = path_family[0][-1][1] + 1

    # Normalized score
    path_family_length = 0
    for n in range(num_path):
        path_family_length = path_family_length + len(path_family[n])
    score_n = (score - M) / (path_family_length + eps)

    # Normalized coverage
    segment_family, coverage = compute_induced_segment_family_coverage(path_family)
    coverage_n = (coverage - M) / (N + eps)

    # Fitness measure
    fitness = 2 * score_n * coverage_n / (score_n + coverage_n + eps)

    return fitness, score, score_n, coverage, coverage_n, path_family_length

N = S.shape[0]

segment_family, coverage = compute_induced_segment_family_coverage(path_family)
fitness, score, score_n, coverage, coverage_n, path_family_length = compute_fitness(
    path_family, score, N)

print('Segment (alpha):', seg)
print('Length of segment:', seg[-1]-seg[0]+1)
print('Length of feature sequence:', N)
print('Induced segment path family:\n', segment_family)
print('Fitness: %0.3f'%fitness)
print('Score: %0.3f'%score)
print('Normalized score: %0.3f'%score_n)
print('Coverage: %d'%coverage)
print('Normalized coverage: %0.3f'%coverage_n)
print('Length of all paths of family: %d'%path_family_length)

# %% [markdown]
# ## Thumbnail Selection
#
# The **audio thumbnail** is defined to be the segment of maximal fitness.

# %% [markdown]
# ## Examples of Optimal Path Families

# %%
def plot_ssm_ann_optimal_path_family(S, ann, seg, Fs=1, cmap='gray_r', color_ann=[], fontsize=12,
                                     figsize=(5, 4.5), ylabel=''):
    """Plot SSM, annotations, and computed optimal path family

    Notebook: C4/C4S3_AudioThumbnailing.ipynb
    """
    N = S.shape[0]
    S_seg = S[:, seg[0]:seg[1]+1]
    D, score = compute_accumulated_score_matrix(S_seg)
    path_family = compute_optimal_path_family(D)
    fitness, score, score_n, coverage, coverage_n, path_family_length = compute_fitness(
        path_family, score, N)
    title = r'$\bar{\sigma}(\alpha)=%0.2f$, $\bar{\gamma}(\alpha)=%0.2f$, $\varphi(\alpha)=%0.2f$' % \
            (score_n, coverage_n, fitness)
    fig, ax, im = plot_ssm_ann(S, ann, color_ann=color_ann, Fs=Fs, cmap=cmap,
                               figsize=figsize, fontsize=fontsize,
                               xlabel=r'$\alpha=[%d:%d]$' % (seg[0], seg[-1]), ylabel=ylabel, title=title)
    plot_path_family(ax[0, 0], path_family, Fs=Fs, x_offset=seg[0])
    return fig, ax, im

SP_all = libfmp.c4.compute_fitness_scape_plot(S)
SP = SP_all[0]
seg = libfmp.c4.seg_max_sp(SP)
fig, ax, im = plot_ssm_ann_optimal_path_family(S, ann_frames, seg,
                    color_ann=color_ann, cmap=cmap_penalty, ylabel='Time (frames)')
plt.show()
path_family = libfmp.c4.check_segment(seg, S)

# %%
seg = [84, 137]
fig, ax, im = plot_ssm_ann_optimal_path_family(S, ann_frames, seg,
                color_ann=color_ann, cmap=cmap_penalty, ylabel='Time (frames)')

# %%
seg = [260, 306]
fig, ax, im = plot_ssm_ann_optimal_path_family(S, ann_frames, seg,
                color_ann=color_ann, cmap=cmap_penalty, ylabel='Time (frames)')
plt.show()
path_family = libfmp.c4.check_segment(seg, S)

# %%
seg = [41, 180]
fig, ax, im = plot_ssm_ann_optimal_path_family(S, ann_frames, seg,
                color_ann=color_ann, cmap=cmap_penalty, ylabel='Time (frames)')
plt.show()
path_family = libfmp.c4.check_segment(seg, S)

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Angel Villar-Corrales.

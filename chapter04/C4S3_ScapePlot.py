# %% [markdown]
# # Scape Plot Representation
#
# Following Section 4.3.2 of [Müller, FMP, Springer 2015], we introduce in this notebook
# the concept of scape plots and apply them for visualizing the fitness of segments.

# %% [markdown]
# ## Triangular Representation of Segments
#
# Each segment can be uniquely described by its center and its length. Using the center
# to parameterize a horizontal axis and the length to parameterize the height, each
# segment can be represented by a point in a **triangular representation**.

# %% [markdown]
# ## Scape Plot
#
# The triangular representation can be used as a grid for visualizing a specific
# numeric property that can be computed for all segments.

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
import libfmp.c4
from libfmp.b import FloatingBox

%matplotlib inline

def visualize_scape_plot(SP, Fs=1, ax=None, figsize=(4, 3), title='',
                         xlabel='Center (seconds)', ylabel='Length (seconds)', interpolation='nearest'):
    """Visualize scape plot

    Notebook: C4/C4S3_ScapePlot.ipynb
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    N = SP.shape[0]
    SP_vis = np.zeros((N, N))
    for length_minus_one in range(N):
        for start in range(N-length_minus_one):
            center = start + length_minus_one//2
            SP_vis[length_minus_one, center] = SP[length_minus_one, start]

    extent = np.array([-0.5, (N-1)+0.5, -0.5, (N-1)+0.5]) / Fs
    im = plt.imshow(SP_vis, cmap='hot_r', aspect='auto', origin='lower', extent=extent, interpolation=interpolation)
    x = np.asarray(range(N))
    x_half_lower = x/2
    x_half_upper = x/2 + N/2 - 1/2
    plt.plot(x_half_lower/Fs, x/Fs, '-', linewidth=3, color='black')
    plt.plot(x_half_upper/Fs, np.flip(x, axis=0)/Fs, '-', linewidth=3, color='black')
    plt.plot(x/Fs, np.zeros(N)/Fs, '-', linewidth=3, color='black')
    plt.xlim([0, (N-1) / Fs])
    plt.ylim([0, (N-1) / Fs])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.colorbar(im, ax=ax)
    return fig, ax, im

N = 9
SP = np.zeros((N,N))
for k in range(N):
    for s in range(N-k):
        length = k + 1
        SP[k, s]= length/N

plt.figure(figsize=(7,3))
ax = plt.subplot(121)
plt.imshow(SP, cmap='hot_r', aspect='auto')
ax.set_title('Data structure (N = %d)'%N)
ax.set_xlabel('Segment start (samples)')
ax.set_ylabel('Length minus one (samples)')
plt.colorbar()

ax = plt.subplot(122)
fig, ax, im = visualize_scape_plot(SP, Fs=1, ax=ax, title='Scape plot visualization',
                xlabel='Segment center (samples)', ylabel='Length minus one (samples)')

# %% [markdown]
# ## Fitness Scape Plot
#
# We now use the scape plot representation for visualizing the fitness measure for all
# segments.

# %%
fn_wav = os.path.join('..', 'data', 'C4', 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.wav')
tempo_rel_set = libfmp.c4.compute_tempo_rel_set(0.66, 1.5, 5)
penalty = -2
x, x_duration, X, Fs_feature, S, I = libfmp.c4.compute_sm_from_filename(fn_wav, L=41, H=10,
                        L_smooth=8, tempo_rel_set=tempo_rel_set, penalty=penalty, thresh= 0.15)
S = libfmp.c4.normalization_properties_ssm(S)

fn_ann_color = 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.csv'
fn_ann = os.path.join('..', 'data', 'C4', fn_ann_color)
ann_frames, color_ann = libfmp.c4.read_structure_annotation(fn_ann, fn_ann_color, Fs=Fs_feature)

cmap_penalty = libfmp.c4.colormap_penalty(penalty=penalty)
fig, ax, im = libfmp.c4.plot_ssm_ann(S, ann_frames, Fs=1, color_ann=color_ann, cmap=cmap_penalty,
                       xlabel='Time (frames)', ylabel='Time (frames)')

# %%
def compute_fitness_scape_plot(S):
    """Compute scape plot for fitness and other measures

    Notebook: C4/C4S3_ScapePlot.ipynb
    """
    N = S.shape[0]
    SP_fitness = np.zeros((N, N))
    SP_score = np.zeros((N, N))
    SP_score_n = np.zeros((N, N))
    SP_coverage = np.zeros((N, N))
    SP_coverage_n = np.zeros((N, N))

    for length_minus_one in range(N):
        for start in range(N-length_minus_one):
            S_seg = S[:, start:start+length_minus_one+1]
            D, score = libfmp.c4.compute_accumulated_score_matrix(S_seg)
            path_family = libfmp.c4.compute_optimal_path_family(D)
            fitness, score, score_n, coverage, coverage_n, path_family_length = libfmp.c4.compute_fitness(
                path_family, score, N)
            SP_fitness[length_minus_one, start] = fitness
            SP_score[length_minus_one, start] = score
            SP_score_n[length_minus_one, start] = score_n
            SP_coverage[length_minus_one, start] = coverage
            SP_coverage_n[length_minus_one, start] = coverage_n
    SP_all = [SP_fitness, SP_score, SP_score_n, SP_coverage, SP_coverage_n]
    return SP_all

SP_all = compute_fitness_scape_plot(S)

# %%
def seg_max_sp(SP):
    """Return segment with maximal value in SP

    Notebook: C4/C4S3_ScapePlot.ipynb
    """
    N = SP.shape[0]
    arg_max = np.argmax(SP)
    ind_max = np.unravel_index(arg_max, [N, N])
    seg = [ind_max[1], ind_max[1]+ind_max[0]]
    return seg

def plot_seg_in_sp(ax, seg, S=None, Fs=1):
    """Plot segment and induced segments as points in SP visualization

    Notebook: C4/C4S3_ScapePlot.ipynb
    """
    if S is not None:
        S_seg = S[:, seg[0]:seg[1]+1]
        D, score = libfmp.c4.compute_accumulated_score_matrix(S_seg)
        path_family = libfmp.c4.compute_optimal_path_family(D)
        segment_family, coverage = libfmp.c4.compute_induced_segment_family_coverage(path_family)
        length = segment_family[:, 1] - segment_family[:, 0] + 1
        center = segment_family[:, 0] + length//2
        ax.scatter(center/Fs, length/Fs, s=64, c='white', zorder=9999)
        ax.scatter(center/Fs, length/Fs, s=16, c='lime', zorder=9999)
    length = seg[1] - seg[0] + 1
    center = seg[0] + length//2
    ax.scatter(center/Fs, length/Fs, s=64, c='white', zorder=9999)
    ax.scatter(center/Fs, length/Fs, s=16, c='blue', zorder=9999)

def plot_sp_ssm(SP, seg, S, ann, color_ann=[], title='', figsize=(5, 4)):
    """Visualization of SP and SSM

    Notebook: C4/C4S3_ScapePlot.ipynb
    """
    float_box = libfmp.b.FloatingBox()
    fig, ax, im = visualize_scape_plot(SP, figsize=figsize, title=title,
                                       xlabel='Center (frames)', ylabel='Length (frames)')
    plot_seg_in_sp(ax, seg, S)
    float_box.add_fig(fig)

    penalty = np.min(S)
    cmap_penalty = libfmp.c4.colormap_penalty(penalty=penalty)
    fig, ax, im = libfmp.c4.plot_ssm_ann_optimal_path_family(
        S, ann, seg, color_ann=color_ann, fontsize=8, cmap=cmap_penalty, figsize=(4, 4),
        ylabel='Time (frames)')
    float_box.add_fig(fig)
    float_box.show()

def check_segment(seg, S):
    """Prints properties of segments with regard to SSM S

    Notebook: C4/C4S3_ScapePlot.ipynb
    """
    N = S.shape[0]
    S_seg = S[:, seg[0]:seg[1]+1]
    D, score = libfmp.c4.compute_accumulated_score_matrix(S_seg)
    path_family = libfmp.c4.compute_optimal_path_family(D)
    fitness, score, score_n, coverage, coverage_n, path_family_length = libfmp.c4.compute_fitness(
                path_family, score, N)
    segment_family, coverage2 = libfmp.c4.compute_induced_segment_family_coverage(path_family)
    print('Segment (alpha):', seg)
    print('Length of segment:', seg[-1]-seg[0]+1)
    print('Length of feature sequence:', N)
    print('Induced segment path family:\n', segment_family)
    print('Fitness: %0.10f' % fitness)
    print('Score: %0.10f' % score)
    print('Normalized score: %0.10f' % score_n)
    print('Coverage: %d, %d' % (coverage, coverage2))
    print('Normalized coverage: %0.10f' % coverage_n)
    print('Length of all paths of family: %d' % path_family_length)
    return path_family

figsize=(5,4)
SP = SP_all[0]
seg = seg_max_sp(SP)
plot_sp_ssm(SP=SP, seg=seg, S=S, ann=ann_frames, color_ann=color_ann,
            title='Scape plot: Fitness', figsize=figsize)
plt.show()
path_family = check_segment(seg, S)

# %% [markdown]
# ## Normalized Score and Coverage
#
# In the definition of the fitness measure, the normalization of score and coverage
# as well as the combination (harmonic mean) of the two measures is of crucial importance.

# %%
SP = SP_all[1]
seg = seg_max_sp(SP)
plot_sp_ssm(SP=SP, seg=seg, S=S, ann=ann_frames, color_ann=color_ann,
            title='Scape plot: Score', figsize=figsize)
path_family = check_segment(seg, S)

# %%
SP = SP_all[2]
seg = seg_max_sp(SP)
plot_sp_ssm(SP=SP, seg=seg, S=S, ann=ann_frames, color_ann=color_ann,
            title='Scape plot: Normalized score', figsize=figsize)
path_family = check_segment(seg, S)

# %%
SP = SP_all[3]
seg = seg_max_sp(SP)
plot_sp_ssm(SP=SP, seg=seg, S=S, ann=ann_frames, color_ann=color_ann,
            title='Scape plot: Coverage', figsize=figsize)
path_family = check_segment(seg, S)

# %%
SP = SP_all[4]
seg = seg_max_sp(SP)
plot_sp_ssm(SP=SP, seg=seg, S=S, ann=ann_frames, color_ann=color_ann,
            title='Scape plot: Normalized coverage', figsize=figsize)
path_family = check_segment(seg, S)

# %% [markdown]
# ## Example: Beatles Songs

# %%
fn_ann_color = 'FMP_C4_Audio_Beatles_YouCantDoThat.csv'
fn_ann = os.path.join('..', 'data', 'C4', fn_ann_color)
fn_wav = os.path.join('..', 'data', 'C4', 'FMP_C4_Audio_Beatles_YouCantDoThat.wav')

tempo_rel_set = libfmp.c4.compute_tempo_rel_set(0.66, 1.5, 5)
penalty = -2
x, x_duration, X, Fs_feature, S, I = libfmp.c4.compute_sm_from_filename(fn_wav, L=21, H=10,
                        L_smooth=8, tempo_rel_set=tempo_rel_set, penalty=penalty, thresh= 0.15)
S = libfmp.c4.normalization_properties_ssm(S)

ann_frames, color_ann = libfmp.c4.read_structure_annotation(fn_ann, fn_ann_color, Fs=Fs_feature)
color_ann = {'I': [1, 0, 0, 0.2], 'V': [0, 1, 0, 0.2], 'B': [0, 0, 1, 0.2], '': [1, 1, 1, 0.2]}

cmap_penalty = libfmp.c4.colormap_penalty(penalty=penalty)
fig, ax, im = libfmp.c4.plot_ssm_ann(S, ann_frames, Fs=1, color_ann=color_ann, cmap=cmap_penalty,
                       xlabel='Time (frames)', ylabel='Time (frames)')

# %%
SP_all = compute_fitness_scape_plot(S)
figsize=(5,4)
SP = SP_all[0]
seg = seg_max_sp(SP)
plot_sp_ssm(SP=SP, seg=seg, S=S, ann=ann_frames, color_ann=color_ann,
            title='Fitness scape plot', figsize=figsize)
path_family = check_segment(seg, S)

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Angel Villar-Corrales.

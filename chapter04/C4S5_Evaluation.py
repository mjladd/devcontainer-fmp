# %% [markdown]
# # Evaluation
#
# Following Section 4.5 of [Müller, FMP, Springer 2015], we discuss some general
# evaluation metrics that are the basis for evaluating music processing tasks.

# %% [markdown]
# ## Introduction
#
# A general evaluation approach is to compare an **estimated result** obtained by some
# automated procedure against some **reference result**.

# %%
import os, sys, librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.append('..')
import libfmp.b
from libfmp.b import FloatingBox
import libfmp.c4
%matplotlib inline

ann_Brahms = {}
ann_Brahms[0] = [[0, 16, 'G major'], [16, 28, 'G minor'], [28, 40, 'G major']]

ann_Brahms[1] = [[0, 4, 'A'], [4, 8, 'A'],  [8, 12, 'B'], [12, 16, 'B'],
                [16, 27, 'C'], [27, 32, 'A'], [32, 36, 'B'], [36, 39, 'B'], [39, 40, '']]

ann_Brahms[2] = [[0, 2, 'a'], [2, 4, 'a'], [4, 6, 'a'], [6, 8, 'a'],
                [8, 10, 'b'], [10, 12, 'c'], [12, 13, 'b'], [13, 15, 'c'],
                [15, 18, 'd'], [18, 20, 'd'],
                [20, 22, 'e'], [22, 24, 'e'], [24, 26, 'e'], [26, 28, 'e'],
                [28, 30, 'a'], [30, 32, 'a'],
                [32,34, 'b'], [34, 36, 'c'], [36, 37, 'b'], [37, 39, 'c'], [39, 40, '']]

color_ann_Brahms = {'G major': [1, 0.8, 0, 0.3], 'G minor': [0, 0, 1, 0.3],
    'A': [1, 0, 0, 0.3], 'B': [0, 1, 0, 0.3], 'C': [0, 0, 1, 0.3], '': [1, 1, 1, 0.3],
    'a': [1, 0, 0, 0.3], 'b': [0, 1, 0, 0.2], 'c': [0.3, 0.5, 0, 0.6],
    'd': [0, 0, 1, 0.2], 'e': [0.2, 0, 0.5, 0.6],  '': [1, 1, 1, 1]}

figsize = (6,1)
for k in range(3):
    libfmp.b.plot_segments(ann_Brahms[k], figsize = figsize, colors=color_ann_Brahms);

# %% [markdown]
# ## Precision, Recall, F-Measure
#
# Many evaluation measures are based on some notion of precision, recall, and
# F-measure--a concept that has been borrowed from the fields of information retrieval
# and pattern recognition.

# %%
def measure_prf(num_TP, num_FN, num_FP):
    """Compute P, R, and F from size of TP, FN, and FP [FMP, Section 4.5.1]

    Notebook: C4/C4S5_Evaluation.ipynb
    """
    P = num_TP / (num_TP + num_FP)
    R = num_TP / (num_TP + num_FN)
    if (P + R) > 0:
        F = 2 * P * R / (P + R)
    else:
        F = 0
    return P, R, F


def measure_prf_sets(I, I_ref_pos, I_est_pos, details=False):
    """Compute P, R, and F from sets I, I_ref_pos, I_est_pos [FMP, Section 4.5.1]

    Notebook: C4/C4S5_Evaluation.ipynb
    """
    I_ref_neg = I.difference(I_ref_pos)
    I_est_neg = I.difference(I_est_pos)
    TP = I_est_pos.intersection(I_ref_pos)
    FN = I_est_neg.intersection(I_ref_pos)
    FP = I_est_pos.intersection(I_ref_neg)
    P, R, F = measure_prf(len(TP), len(FN), len(FP))
    if details:
        print('TP = ', TP, ';  FN = ', FN, ';  FP = ', FP)
        print('P = %0.3f;  R = %0.3f;  F = %0.3f' % (P, R, F))
    return P, R, F

print('Example 1')
I = {0,1,2,3,4,5,6,7,8,9}
I_ref_pos = {0,1,2,3}
I_est_pos = {0,1,2,4,5}
P, R, F = measure_prf_sets(I, I_ref_pos, I_est_pos, details=True)

# %%
print('Example 2')
I = {0,1,2,3,4,5,6,7,8,9}
I_ref_pos = {0,1,2,3,4,5,6}
I_est_pos = {0,1,2,7,8,9}
P, R, F = measure_prf_sets(I, I_ref_pos, I_est_pos, details=True)

# %%
print('Example 3')
I = {0,1,2,3,4,5,6,7,8,9}
I_ref_pos = I
I_est_pos = {0,1,2}
P, R, F = measure_prf_sets(I, I_ref_pos, I_est_pos, details=True)

# %% [markdown]
# ## Labeling Evaluation
#
# We now introduce some frame-based evaluation measures: **pairwise precision**,
# **pairwise recall**, and **pairwise F-measure**.

# %%
def convert_ann_to_seq_label(ann):
    """Convert structure annotation with integer time positions into label sequence

    Notebook: C4/C4S5_Evaluation.ipynb
    """
    X = []
    for seg in ann:
        K = seg[1] - seg[0]
        for k in range(K):
            X.append(seg[2])
    return X

def plot_seq_label(ax, X, Fs=1, color_label=[], direction='horizontal',
                   fontsize=10, time_axis=False, print_labels=True):
    """Plot label sequence in the style of annotations

    Notebook: C4/C4S5_Evaluation.ipynb
    """
    ann_X = []
    for m, cur_x in enumerate(X):
        ann_X.append([(m-0.5)/Fs, (m+0.5)/Fs, cur_x])
    libfmp.b.plot_segments(ann_X, ax=ax, time_axis=time_axis, fontsize=fontsize,
                           direction=direction, colors=color_label, print_labels=print_labels)
    return ann_X

color_label = {'A': [1, 0, 0, 0.2], 'B': [0, 0, 1, 0.2], 'C': [0, 1, 0, 0.2],
             'X': [1, 0, 0, 0.2], 'Y': [0, 0, 1, 0.2], 'Z': [0, 1, 0, 0.2]}

ann_ref = [[0, 4, 'A'], [4, 7, 'B'], [7, 10, 'A']]
ann_est = [[0, 1, 'X'], [1, 3, 'Y'], [3, 7, 'Z'], [7, 9, 'Y'], [9, 10, 'X']]

X_ref = convert_ann_to_seq_label(ann_ref)
X_est = convert_ann_to_seq_label(ann_est)

print('Segment-based structure annotation:')
plt.figure(figsize=(5,2))
ax = plt.subplot(211)
libfmp.b.plot_segments(ann_ref, ax=ax, colors=color_label);
ax = plt.subplot(212)
libfmp.b.plot_segments(ann_est, ax=ax, colors=color_label);
plt.tight_layout()
plt.show()

print('Frame-based label sequence:')
plt.figure(figsize=(5,2))
ax = plt.subplot(211)
plot_seq_label(ax, X_ref, color_label=color_label, time_axis=True);
ax = plt.subplot(212)
plot_seq_label(ax, X_est, color_label=color_label, time_axis=True);
plt.tight_layout()

# %%
def compare_pairwise(X):
    """Compute set of positive items from label sequence [FMP, Section 4.5.3]

    Notebook: C4/C4S5_Evaluation.ipynb
    """
    N = len(X)
    I_pos = np.zeros((N, N))
    for n in range(1, N):
        for m in range(n):
            if X[n] is X[m]:
                I_pos[n, m] = 1
    return I_pos

def evaluate_pairwise(I_ref_pos, I_est_pos):
    """Compute pairwise evaluation measures [FMP, Section 4.5.3]

    Notebook: C4/C4S5_Evaluation.ipynb
    """
    I_eval = np.zeros(I_ref_pos.shape)
    TP = (I_ref_pos + I_est_pos) > 1
    FN = (I_ref_pos - I_est_pos) > 0
    FP = (I_ref_pos - I_est_pos) < 0
    I_eval[TP] = 1
    I_eval[FN] = 2
    I_eval[FP] = 3
    num_TP = np.sum(TP)
    num_FN = np.sum(FN)
    num_FP = np.sum(FP)
    P, R, F = measure_prf(num_TP, num_FN, num_FP)
    return P, R, F, num_TP, num_FN, num_FP, I_eval

def plot_matrix_label(M, X, color_label=None, figsize=(3, 3), cmap='gray_r', fontsize=8, print_labels=True):
    """Plot matrix and label sequence

    Notebook: C4/C4S5_Evaluation.ipynb
    """
    fig, ax = plt.subplots(2, 3, gridspec_kw={'width_ratios': [0.1, 1, 0.05],
                                              'wspace': 0.2, 'height_ratios': [1, 0.1]},
                           figsize=figsize)

    colorList = np.array([[1, 1, 1, 1],  [0, 0, 0, 0.7]])
    cmap = ListedColormap(colorList)
    im = ax[0, 1].imshow(M, aspect='auto', cmap=cmap,  origin='lower', interpolation='nearest')
    im.set_clim(vmin=-0.5, vmax=1.5)
    ax_cb = plt.colorbar(im, cax=ax[0, 2])
    ax_cb.set_ticks(np.arange(0, 2, 1))
    ax_cb.set_ticklabels(np.arange(0, 2, 1))
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    plot_seq_label(ax[1, 1], X, color_label=color_label, fontsize=fontsize, print_labels=print_labels)
    ax[1, 2].axis('off')
    ax[1, 0].axis('off')
    plot_seq_label(ax[0, 0], X, color_label=color_label, fontsize=fontsize, print_labels=print_labels,
                   direction='vertical')
    return fig, ax

def plot_matrix_pairwise(I_eval, figsize=(3, 2.5)):
    """Plot matrix I_eval encoding TP, FN, FP

    Notebook: C4/C4S5_Evaluation.ipynb
    """
    fig = plt.figure(figsize=figsize)
    colorList = np.array([[1, 1, 1, 1], [0, 0.7, 0, 1], [1, 0, 0, 1], [1, 0.5, 0.5, 1]])
    cmap = ListedColormap(colorList)
    im = plt.imshow(I_eval, aspect='auto', cmap=cmap,  origin='lower', interpolation='nearest')
    im.set_clim(vmin=-0.5, vmax=3.5)
    plt.xticks([])
    plt.yticks([])
    ax_cb = plt.colorbar(im)
    ax_cb.set_ticks(np.arange(0, 4, 1))
    ax_cb.set_ticklabels(['', 'TP', 'FN', 'FP'])
    return fig, im

float_box = libfmp.b.FloatingBox(align='top')
I_ref_pos = compare_pairwise(X_ref)
fig, ax = plot_matrix_label(I_ref_pos, X_ref, color_label=color_label)
ax[0,1].set_title('Reference')
float_box.add_fig(fig)

I_est_pos = compare_pairwise(X_est)
fig, ax = plot_matrix_label(I_est_pos, X_est, color_label=color_label)
ax[0,1].set_title('Estimation')
float_box.add_fig(fig)

P, R, F, num_TP, num_FN, num_FP, I_eval =  evaluate_pairwise(I_ref_pos, I_est_pos)
fig, im = plot_matrix_pairwise(I_eval)
plt.title('Pairwise TP, FN, FP')
float_box.add_fig(fig)
float_box.show()

print('#TP = ', num_TP, ';  #FN = ', num_FN, ';  #FP = ', num_FP)
print('P = %0.3f;  R = %0.3f;  F = %0.3f' % (P, R, F))

# %%
X_0 = convert_ann_to_seq_label(ann_Brahms[0])
X_1 = convert_ann_to_seq_label(ann_Brahms[1])
X_2 = convert_ann_to_seq_label(ann_Brahms[2])

X_set = [X_0, X_1, X_2]
combinations = [(0,1), (0,2), (1,2)]
case_label = ['Coarse', 'Medium', 'Fine']

for c in combinations:
    X_ref = X_set[c[0]]
    X_est = X_set[c[1]]

    float_box = libfmp.b.FloatingBox(align='top')
    I_ref_pos = compare_pairwise(X_ref)
    fig, ax = plot_matrix_label(I_ref_pos, X_ref, color_label=color_ann_Brahms, print_labels=False)
    ax[0,1].set_title('Reference: '+case_label[c[0]])
    float_box.add_fig(fig)

    I_est_pos = compare_pairwise(X_est)
    fig, ax = plot_matrix_label(I_est_pos, X_est, color_label=color_ann_Brahms, print_labels=False)
    ax[0,1].set_title('Estimation: '+case_label[c[1]])
    float_box.add_fig(fig)

    P, R, F, num_TP, num_FN, num_FP, I_eval =  evaluate_pairwise(I_ref_pos, I_est_pos)
    fig, im = plot_matrix_pairwise(I_eval)
    plt.title('Pairwise TP, FN, FP')
    float_box.add_fig(fig)

    float_box.show()

    print('#TP = ', num_TP, ';  #FN = ', num_FN, ';  #FP = ', num_FP)
    print('P = %0.3f;  R = %0.3f;  F = %0.3f' % (P, R, F))

# %% [markdown]
# ## Boundary Evaluation
#
# The pairwise precision, recall, and F-measure are solely based on label information,
# whereas segment boundaries are treated implicitly. For novelty-based segmentation,
# the precise detection of boundaries is the focus.

# %%
def evaluate_boundary(B_ref, B_est, tau):
    """Compute boundary evaluation measures [FMP, Section 4.5.4]

    Notebook: C4/C4S5_Evaluation.ipynb
    """
    N = len(B_ref)
    num_TP = 0
    num_FN = 0
    num_FP = 0
    B_tol = np.zeros((np.array([B_ref])).shape)
    B_eval = np.zeros((np.array([B_ref])).shape)
    for n in range(N):
        min_idx = max(0, n - tau)
        max_idx = min(N - 1, n + tau)
        if B_ref[n] == 1:
            B_tol[:, min_idx:max_idx+1] = 2
            B_tol[:, n] = 1
            temp = sum(B_est[min_idx:max_idx+1])
            if temp > 0:
                num_TP += temp
            else:
                num_FN += 1
                B_eval[:, n] = 2
        if B_est[n] == 1:
            if sum(B_ref[min_idx:max_idx+1]) == 0:
                num_FP += 1
                B_eval[:, n] = 3
            else:
                B_eval[:, n] = 1
    P, R, F = measure_prf(num_TP, num_FN, num_FP)
    return P, R, F, num_TP, num_FN, num_FP, B_tol, B_eval

def plot_boundary_measures(B_ref, B_est, tau, figsize=(8, 2.5)):
    """Plot B_ref and B_est

    Notebook: C4/C4S5_Evaluation.ipynb
    """
    P, R, F, num_TP, num_FN, num_FP, B_tol, B_eval = evaluate_boundary(B_ref, B_est, tau)

    colorList = np.array([[1., 1., 1., 1.], [0., 0., 0., 1.], [0.7, 0.7, 0.7, 1.]])
    cmap_tol = ListedColormap(colorList)
    colorList = np.array([[1, 1, 1, 1], [0, 0.7, 0, 1], [1, 0, 0, 1], [1, 0.5, 0.5, 1]])
    cmap_measures = ListedColormap(colorList)

    fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.02],
                                              'wspace': 0.2, 'height_ratios': [1, 1, 1]},
                           figsize=figsize)

    im = ax[0, 0].imshow(B_tol, cmap=cmap_tol, interpolation='nearest')
    ax[0, 0].set_title('Reference boundaries (with tolerance)')
    im.set_clim(vmin=-0.5, vmax=2.5)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax_cb = plt.colorbar(im, cax=ax[0, 1])
    ax_cb.set_ticks(np.arange(0, 3, 1))
    ax_cb.set_ticklabels(['', 'Positive', 'Tolerance'])

    im = ax[1, 0].imshow(np.array([B_est]), cmap=cmap_tol, interpolation='nearest')
    ax[1, 0].set_title('Estimated boundaries')
    im.set_clim(vmin=-0.5, vmax=2.5)
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax_cb = plt.colorbar(im, cax=ax[1, 1])
    ax_cb.set_ticks(np.arange(0, 3, 1))
    ax_cb.set_ticklabels(['', 'Positive', 'Tolerance'])

    im = ax[2, 0].imshow(B_eval, cmap=cmap_measures, interpolation='nearest')
    ax[2, 0].set_title('Evaluation')
    im.set_clim(vmin=-0.5, vmax=3.5)
    ax[2, 0].set_xticks([])
    ax[2, 0].set_yticks([])
    ax_cb = plt.colorbar(im, cax=ax[2, 1])
    ax_cb.set_ticks(np.arange(0, 4, 1))
    ax_cb.set_ticklabels(['', 'TP', 'FN', 'FP'])
    plt.show()
    return fig, ax


B_ref = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
B_est = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
tau_list = [0,1,2]

for tau in tau_list:
    print('====== Evaluation using tau = %d ======'%tau)
    P, R, F, num_TP, num_FN, num_FP, B_tol, B_eval = evaluate_boundary(B_ref, B_est, tau)
    print('#TP = ', num_TP, ';  #FN = ', num_FN, ';  #FP = ', num_FP)
    print('P = %0.3f;  R = %0.3f;  F = %0.3f' % (P, R, F))
    fig, ax = plot_boundary_measures(B_ref, B_est, tau=tau)

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Tim Zunner.

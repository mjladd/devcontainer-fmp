# %% [markdown]
# # DTW Variants
#
# Various modifications have been proposed in order to speed up DTW computations as well
# as to better control the overall course of the warping paths. Following Section 3.2.2
# of [Müller, FMP, Springer 2015], we discuss in this notebook some of these DTW variants.

# %% [markdown]
# ## Step Size Condition
#
# In classical DTW, the step size condition is expressed by the set Sigma={(1,0),(0,1),(1,1)}.
# One drawback of this condition is that a single element of one sequence may be assigned
# to many consecutive elements of the other sequence, which leads to vertical and horizontal
# sections in the warping path.

# %% [markdown]
# ## libfmp Implementation
#
# In the next code cell, we provide an implementation the DTW algorithm with step size
# condition Sigma={(1,1),(2,1),(1,2)}.

# %%
import numpy as np
import scipy.spatial
import librosa
import matplotlib.pyplot as plt
from numba import jit
%matplotlib inline

import sys
sys.path.append('..')
import libfmp.c3

@jit(nopython=True)
def compute_accumulated_cost_matrix_21(C):
    """Compute the accumulated cost matrix given the cost matrix

    Notebook: C3/C3S2_DTWvariants.ipynb

    Args:
        C (np.ndarray): Cost matrix

    Returns:
        D (np.ndarray): Accumulated cost matrix
    """
    N = C.shape[0]
    M = C.shape[1]
    D = np.zeros((N + 2, M + 2))
    D[:, 0:2] = np.inf
    D[0:2, :] = np.inf
    D[2, 2] = C[0, 0]

    for n in range(N):
        for m in range(M):
            if n == 0 and m == 0:
                continue
            D[n+2, m+2] = C[n, m] + min(D[n-1+2, m-1+2], D[n-2+2, m-1+2], D[n-1+2, m-2+2])
    D = D[2:, 2:]
    return D

@jit(nopython=True)
def compute_optimal_warping_path_21(D):
    """Compute the warping path given an accumulated cost matrix

    Notebook: C3/C3S2_DTWvariants.ipynb

    Args:
        D (np.ndarray): Accumulated cost matrix

    Returns:
        P (np.ndarray): Optimal warping path
    """
    N = D.shape[0]
    M = D.shape[1]
    n = N - 1
    m = M - 1
    P = [(n, m)]
    while n > 0 or m > 0:
        if n == 0:
            cell = (0, m - 1)
        elif m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n-1, m-1], D[n-2, m-1], D[n-1, m-2])
            if val == D[n-1, m-1]:
                cell = (n-1, m-1)
            elif val == D[n-2, m-1]:
                cell = (n-2, m-1)
            else:
                cell = (n-1, m-2)
        P.append(cell)
        (n, m) = cell
    P.reverse()
    P = np.array(P)
    return P

# Sequences
X = [1, 3, 9, 2, 1]
Y = [2, 0, 0, 8, 7, 2]
N, M = len(X), len(Y)

C = libfmp.c3.compute_cost_matrix(X, Y, metric='euclidean')
D = compute_accumulated_cost_matrix_21(C)
P = compute_optimal_warping_path_21(D)

plt.figure(figsize=(6, 2))
plt.plot(X, c='k', label='X')
plt.plot(Y, c='b', label='Y')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(9, 3))
ax = plt.subplot(1, 2, 1)
libfmp.c3.plot_matrix_with_points(C, P, linestyle='-',
    ax=[ax], aspect='equal', clim=[0, np.max(C)],
    title='$C$ with optimal warping path', xlabel='Sequence Y', ylabel='Sequence X');

ax = plt.subplot(1, 2, 2)
D_max = np.nanmax(D[D != np.inf])
libfmp.c3.plot_matrix_with_points(D, P, linestyle='-',
    ax=[ax], aspect='equal', clim=[0, D_max],
    title='$D$ with optimal warping path', xlabel='Sequence Y', ylabel='Sequence X');
for x, y in zip(*np.where(np.isinf(D))):
    plt.text(y, x, '$\\infty$', horizontalalignment='center', verticalalignment='center')

plt.tight_layout()

# %% [markdown]
# ## LibROSA Implementation
#
# The DTW function of LibROSA allows for specifying arbitrary step size conditions.

# %%
def compute_plot_D_P(X, Y, ax, step_sizes_sigma=np.array([[1, 1], [0, 1], [1, 0]]),
                     weights_mul=np.array([1, 1, 1]), title='',
                     global_constraints=False, band_rad=0.25):
    D, P = librosa.sequence.dtw(X, Y, metric='euclidean', weights_mul=weights_mul,
                    step_sizes_sigma=step_sizes_sigma,
                    global_constraints=global_constraints, band_rad=band_rad)
    D_max = np.nanmax(D[D != np.inf])
    libfmp.c3.plot_matrix_with_points(D, P, linestyle='-',
        ax=[ax], aspect='equal', clim=[0, D_max],
        title= title, xlabel='Sequence Y', ylabel='Sequence X');
    for x, y in zip(*np.where(np.isinf(D))):
        plt.text(y, x, '$\\infty$', horizontalalignment='center', verticalalignment='center')

X = [1, 3, 9, 2, 1, 3, 9, 9]
Y = [2, 0, 0, 9, 1, 7]

print(r'Accumulated cost matrix and optimal warping path for different step size conditions:')
plt.figure(figsize=(10, 3))

ax = plt.subplot(1, 3, 1)
step_sizes_sigma = np.array([[1, 0], [0, 1], [1, 1]])
title='Step sizes:'+''.join(str(s) for s in step_sizes_sigma)
compute_plot_D_P(X, Y, ax=ax, step_sizes_sigma=step_sizes_sigma, title=title)

ax = plt.subplot(1, 3, 2)
step_sizes_sigma = np.array([[1, 1], [2, 1], [1, 2]])
title='Step sizes:'+''.join(str(s) for s in step_sizes_sigma)
compute_plot_D_P(X, Y, ax=ax, step_sizes_sigma=step_sizes_sigma, title=title)

ax = plt.subplot(1, 3, 3)
step_sizes_sigma = np.array([[1, 1], [3, 1], [1, 3]])
title='Step sizes:'+''.join(str(s) for s in step_sizes_sigma)
compute_plot_D_P(X, Y, ax=ax, step_sizes_sigma=step_sizes_sigma, title=title)

plt.tight_layout()

# %% [markdown]
# ## Local Weights
#
# To favor the vertical, horizontal, or diagonal direction in the alignment, one can
# introduce additional **local weights**.

# %%
X = [1, 1, 1, 1, 2, 1, 1, 6, 6]
Y = [0, 3, 3, 3, 9, 9, 7, 7]

print(r'Accumulated cost matrix and optimal warping path for different local weights:')
plt.figure(figsize=(10, 3))

ax = plt.subplot(1, 3, 1)
weights_mul = np.array([1, 1, 1])
title='Weights: '+'[ '+''.join(str(s)+' ' for s in weights_mul)+']'
compute_plot_D_P(X, Y, ax=ax, weights_mul=weights_mul, title=title)

ax = plt.subplot(1, 3, 2)
weights_mul = np.array([2, 1, 1])
title='Weights: '+'[ '+''.join(str(s)+' ' for s in weights_mul)+']'
compute_plot_D_P(X, Y, ax=ax, weights_mul=weights_mul, title=title)

ax = plt.subplot(1, 3, 3)
weights_mul = np.array([1, 3, 3])
title='Weights: '+'[ '+''.join(str(s)+' ' for s in weights_mul)+']'
compute_plot_D_P(X, Y, ax=ax, weights_mul=weights_mul, title=title)

plt.tight_layout()

# %% [markdown]
# ## Global Constraints
#
# One common DTW variant is to impose **global constraints** on the admissible warping
# paths. Such constraints not only speed up DTW computations but also prevent "pathological"
# alignments by globally controlling the overall course of a warping path.

# %%
X = [1, 1, 1, 1, 2, 1, 1, 6, 6]
Y = [0, 3, 3, 3, 9, 9, 7, 7]

print(r'Accumulated cost matrix and optimal warping path for different constraint regions:')
plt.figure(figsize=(10, 3))
global_constraints = True

ax = plt.subplot(1, 3, 1)
band_rad = 1
title='Sakao-Chiba band (rad = %.2f)'%band_rad
compute_plot_D_P(X, Y, ax=ax, global_constraints=global_constraints, band_rad=band_rad,title=title)

ax = plt.subplot(1, 3, 2)
band_rad = 0.5
title='Sakao-Chiba band (rad = %.2f)'%band_rad
compute_plot_D_P(X, Y, ax=ax, global_constraints=global_constraints, band_rad=band_rad,title=title)

ax = plt.subplot(1, 3, 3)
band_rad = 0.25
title='Sakao-Chiba band (rad = %.2f)'%band_rad
compute_plot_D_P(X, Y, ax=ax, global_constraints=global_constraints, band_rad=band_rad,title=title)

plt.tight_layout()

# %% [markdown]
# ## Multiscale DTW
#
# When using the concept of global constraint regions, one needs to make sure that the
# optimal warping path to be computed actually lies within this region. One possible
# strategy is to use data-dependent constraint regions instead of a data-independent,
# fixed constraint region. This idea can be realized by a **multiscale approach** to DTW.

# %% [markdown]
# ## Further Notes
#
# * The article "An Efficient Multiscale Approach to Audio Synchronization" shows how
#   the multiscale approach can be applied for music synchronization.
# * Extending this work, the article "Memory-Restricted Multiscale Dynamic Time Warping"
#   introduces a DTW approach that works with a pre-defined memory quota.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Frank Zalkow and Meinard Müller.

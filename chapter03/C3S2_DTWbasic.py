# %% [markdown]
# # Dynamic Time Warping (DTW)
#
# Following Section 3.2.1 of [Müller, FMP, Springer 2015], we explain in this notebook
# the basic algorithm for dynamic time warping (DTW).

# %% [markdown]
# ## Basic Idea
#
# Given two sequences X of length N and Y of length M, the objective of **dynamic time
# warping** (DTW) is to temporally align these two sequences in some optimal sense under
# certain constraints.

# %% [markdown]
# ## Warping Path
#
# To model a global alignment between the elements of the sequences X and Y, the idea
# is to consider a sequence of index pairs that fulfills certain constraints. This leads
# to the notion of a **warping path**.

# %% [markdown]
# ## Cost Matrix and Optimality
#
# Next, we introduce a notion that tells us something about the **quality** of a warping
# path. To this end, we need a way to numerically compare the elements of the feature
# sequences X and Y using a **local cost measure**.

# %% [markdown]
# ## DTW Algorithm using Dynamic Programming
#
# To determine an optimal warping path P* for two sequences X and Y, one could compute
# the total cost of all possible (N,M)-warping paths and then take the minimal cost.
# However, the number of different warping paths is exponential in N and M. Therefore,
# such a naive approach is computationally infeasible for large N and M. We now introduce
# an O(NM) algorithm that is based on **dynamic programming**.

# %% [markdown]
# ## Implementation
#
# We now implement the DTW algorithm as described above. As an illustrative example,
# we consider two sequences of real numbers and the absolute value of differences
# (one-dimensional Euclidean distance) as cost measure.

# %%
import numpy as np
import scipy.spatial
from numba import jit
import matplotlib.pyplot as plt
%matplotlib inline

X =  [1, 3, 9, 2, 1]
Y = [2, 0, 0, 8, 7, 2]
N = len(X)
M = len(Y)

plt.figure(figsize=(6, 2))
plt.plot(X, c='k', label='$X$')
plt.plot(Y, c='b', label='$Y$')
plt.legend()
plt.tight_layout()

# %% [markdown]
# We now compute the **cost matrix** C using the Euclidean distance as **local cost measure**.

# %%
def compute_cost_matrix(X, Y, metric='euclidean'):
    """Compute the cost matrix of two feature sequences

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        X (np.ndarray): Sequence 1
        Y (np.ndarray): Sequence 2
        metric (str): Cost metric, a valid strings for scipy.spatial.distance.cdist (Default value = 'euclidean')

    Returns:
        C (np.ndarray): Cost matrix
    """
    X, Y = np.atleast_2d(X, Y)
    C = scipy.spatial.distance.cdist(X.T, Y.T, metric=metric)
    return C

C =  compute_cost_matrix(X, Y, metric='euclidean')
print('Cost matrix C =', C, sep='\n')

# %% [markdown]
# Next, using dynamic programming, we compute the **accumulated cost matrix** D, which
# yields the **DTW distance** DTW(X,Y).

# %%
@jit(nopython=True)
def compute_accumulated_cost_matrix(C):
    """Compute the accumulated cost matrix given the cost matrix

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        C (np.ndarray): Cost matrix

    Returns:
        D (np.ndarray): Accumulated cost matrix
    """
    N = C.shape[0]
    M = C.shape[1]
    D = np.zeros((N, M))
    D[0, 0] = C[0, 0]
    for n in range(1, N):
        D[n, 0] = D[n-1, 0] + C[n, 0]
    for m in range(1, M):
        D[0, m] = D[0, m-1] + C[0, m]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n-1, m], D[n, m-1], D[n-1, m-1])
    return D

D =  compute_accumulated_cost_matrix(C)
print('Accumulated cost matrix D =', D, sep='\n')
print('DTW distance DTW(X, Y) =', D[-1, -1])

# %% [markdown]
# Finally, we derive the optimal warping path P* using backtracking.

# %%
@jit(nopython=True)
def compute_optimal_warping_path(D):
    """Compute the warping path given an accumulated cost matrix

    Notebook: C3/C3S2_DTWbasic.ipynb

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
            val = min(D[n-1, m-1], D[n-1, m], D[n, m-1])
            if val == D[n-1, m-1]:
                cell = (n-1, m-1)
            elif val == D[n-1, m]:
                cell = (n-1, m)
            else:
                cell = (n, m-1)
        P.append(cell)
        (n, m) = cell
    P.reverse()
    return np.array(P)

P = compute_optimal_warping_path(D)
print('Optimal warping path P =', P.tolist())

# %% [markdown]
# As a sanity check, we now compute the **total cost** of the optimal warping path,
# which agrees with DTW(X,Y).

# %%
c_P = sum(C[n, m] for (n, m) in P)
print('Total cost of optimal warping path:', c_P)
print('DTW distance DTW(X, Y) =', D[-1, -1])

# %% [markdown]
# Finally, we visualize the cost matrix C and the accumulated cost matrix D along with
# the optimal warping path (indicated by the red dots).

# %%
P = np.array(P)
plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)
plt.imshow(C, cmap='gray_r', origin='lower', aspect='equal')
plt.plot(P[:, 1], P[:, 0], marker='o', color='r')
plt.clim([0, np.max(C)])
plt.colorbar()
plt.title('$C$ with optimal warping path')
plt.xlabel('Sequence Y')
plt.ylabel('Sequence X')

plt.subplot(1, 2, 2)
plt.imshow(D, cmap='gray_r', origin='lower', aspect='equal')
plt.plot(P[:, 1], P[:, 0], marker='o', color='r')
plt.clim([0, np.max(D)])
plt.colorbar()
plt.title('$D$ with optimal warping path')
plt.xlabel('Sequence Y')
plt.ylabel('Sequence X')

plt.tight_layout()

# %% [markdown]
# ## libfmp Implementation
#
# Some of the above functions involve nested loops, which are inefficient to compute
# when using Python. Using the jit-compiler offered by the Python package Numba, one
# finds accelerated versions of these functions as part of the libfmp-library.

# %%
import sys
sys.path.append('..')

import libfmp.c3

C = libfmp.c3.compute_cost_matrix(X, Y)
D = libfmp.c3.compute_accumulated_cost_matrix(C)
P = libfmp.c3.compute_optimal_warping_path(D)

P = np.array(P)

plt.figure(figsize=(9, 3))
ax = plt.subplot(1, 2, 1)
libfmp.c3.plot_matrix_with_points(C, P, linestyle='-',
    ax=[ax], aspect='equal', clim=[0, np.max(C)],
    title='$C$ with optimal warping path', xlabel='Sequence Y', ylabel='Sequence X');

ax = plt.subplot(1, 2, 2)
libfmp.c3.plot_matrix_with_points(D, P, linestyle='-',
    ax=[ax], aspect='equal', clim=[0, np.max(D)],
    title='$D$ with optimal warping path', xlabel='Sequence Y', ylabel='Sequence X');

plt.tight_layout()

# %% [markdown]
# ## LibROSA Implementation
#
# LibROSA also offers a DTW function that can realize different DTW variants.

# %%
import librosa

D, P = librosa.sequence.dtw(X, Y, metric='euclidean',
                            step_sizes_sigma=np.array([[1, 1], [0, 1], [1, 0]]),
                            weights_add=np.array([0, 0, 0]), weights_mul=np.array([1, 1, 1]))

plt.figure(figsize=(9, 3))
ax = plt.subplot(1, 2, 1)
libfmp.c3.plot_matrix_with_points(C, P, linestyle='-',
    ax=[ax], aspect='equal', clim=[0, np.max(C)],
    title='$C$ with optimal warping path', xlabel='Sequence Y', ylabel='Sequence X');

ax = plt.subplot(1, 2, 2)
libfmp.c3.plot_matrix_with_points(D, P, linestyle='-',
    ax=[ax], aspect='equal', clim=[0, np.max(D)],
    title='$D$ with optimal warping path', xlabel='Sequence Y', ylabel='Sequence X');

plt.tight_layout()

# %% [markdown]
# ## Further Notes
#
# * In the FMP notebook on DTW variants we discuss various modifications to speed up
#   DTW computations as well as to better control the overall course of the warping paths.
# * We represent in the FMP notebook on music synchronization an example application,
#   where we apply DTW for automatically aligning two different recordings of the same
#   piece of music.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Frank Zalkow.

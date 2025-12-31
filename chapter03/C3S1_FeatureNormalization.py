# %% [markdown]
# # Feature Normalization
#
# In this notebook, we introduce different strategies for normalizing a feature
# representation. Parts of the notebook follow Section 3.1.2.1 and Section 2.2.3.3
# of [Müller, FMP, Springer 2015].

# %% [markdown]
# ## Formal Definition of a Norm
#
# Given a vector space (e.g., F = R^K), a norm is a nonnegative function
# p: F -> R_>=0 that satisfies three properties:
#
# - Triangle inequality: p(x + y) <= p(x) + p(y) for all x, y in F
# - Positive scalability: p(alpha * x) = |alpha| * p(x)
# - Positive definiteness: p(x) = 0 if and only if x = 0

# %% [markdown]
# ## Euclidean Norm
#
# The most commonly used norm is the **Euclidean norm** (or l^2-norm):
# ||x||_2 = sqrt(sum(x(k)^2))

# %%
import os
import sys
import numpy as np
import scipy
from matplotlib import pyplot as plt
import librosa
import IPython.display as ipd
from numba import jit

sys.path.append('..')
import libfmp.b

%matplotlib inline


def norm_Euclidean(x):
    p = np.sqrt(np.sum(x ** 2))
    return p


def plot_vector(x, y, color='k', start=0, linestyle='-'):
    return plt.arrow(np.real(start), np.imag(start), x, y,
                     linestyle=linestyle, head_width=0.05,
                     fc=color, ec=color, overhang=0.3, length_includes_head=True)


fig, ax = plt.subplots(figsize=(5, 5))
plt.grid()
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.xlabel('Axis of first coordinate')
plt.ylabel('Axis of second coordinate')

circle = plt.Circle((0, 0), 1, color='r', fill=0, linewidth=2)
ax.add_artist(circle)

x_list = [np.array([[1, 1], [0.6, 1.1]]),
          np.array([[-np.sqrt(2)/2, np.sqrt(2)/2], [-1.45, 0.85]]),
          np.array([[0, -1], [-0.4, -1.2]])]

for y in x_list:
    x = y[0, :]
    p = norm_Euclidean(x)
    color = 'r' if p == 1 else 'k'
    plot_vector(x[0], x[1], color=color)
    plt.text(y[1, 0], y[1, 1], r'$||x||_2=%0.3f$' % p, size='12', color=color)

# %% [markdown]
# ## Manhattan Norm
#
# In the **Manhattan norm** (or l^1-norm), the length of a vector is measured by
# summing up the absolute values of the vector's Cartesian coordinates:
# ||x||_1 = sum(|x(k)|)

# %%
def norm_Manhattan(x):
    p = np.sum(np.abs(x))
    return p


fig, ax = plt.subplots(figsize=(5, 5))
plt.grid()
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.xlabel('Axis of first coordinate')
plt.ylabel('Axis of second coordinate')

plt.plot([-1, 0, 1, 0, -1], [0, 1, 0, -1, 0], color='r', linewidth=2)

for y in x_list:
    x = y[0, :]
    p = norm_Manhattan(x)
    color = 'r' if p == 1 else 'k'
    plot_vector(x[0], x[1], color=color)
    plt.text(y[1, 0], y[1, 1], r'$||x||_1=%0.3f$' % p, size='12', color=color)

# %% [markdown]
# ## Maximum Norm
#
# In the **maximum norm** (or l^infinity-norm), the length of a vector is measured
# by its maximum absolute Cartesian coordinate:
# ||x||_infinity = max(|x(k)|)

# %%
def norm_max(x):
    p = np.max(np.abs(x))
    return p


fig, ax = plt.subplots(figsize=(5, 5))
plt.grid()
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.xlabel('Axis of first coordinate')
plt.ylabel('Axis of second coordinate')

plt.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], color='r', linewidth=2)

for y in x_list:
    x = y[0, :]
    p = norm_max(x)
    color = 'r' if p == 1 else 'k'
    plot_vector(x[0], x[1], color=color)
    plt.text(y[1, 0], y[1, 1], r'$||x||_\infty=%0.3f$' % p, size='12', color=color)

# %% [markdown]
# ## Feature Normalization
#
# To better compare feature representations, one often applies **normalization**.
# One normalization strategy is to choose a suitable norm p and then to replace
# each feature vector x_n by x_n / p(x_n).
#
# As a result, a normalized chroma vector only encodes **relative** rather than
# **absolute** differences in the sizes of the twelve chroma coefficients.
# Normalization introduces a kind of **invariance** to differences in **dynamics**
# or **sound intensity**.

# %% [markdown]
# ## Example: C-Major Scale

# %%
@jit(nopython=True)
def normalize_feature_sequence(X, norm='2', threshold=0.0001, v=None):
    """Normalizes the columns of a feature sequence

    Args:
        X (np.ndarray): Feature sequence
        norm (str): The norm to be applied. '1', '2', 'max' or 'z'
        threshold (float): An threshold below which the vector v used instead
        v (float): Used instead of normalization below threshold

    Returns:
        X_norm (np.ndarray): Normalized feature sequence
    """
    assert norm in ['1', '2', 'max', 'z']

    K, N = X.shape
    X_norm = np.zeros((K, N))

    if norm == '1':
        if v is None:
            v = np.ones(K, dtype=np.float64) / K
        for n in range(N):
            s = np.sum(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == '2':
        if v is None:
            v = np.ones(K, dtype=np.float64) / np.sqrt(K)
        for n in range(N):
            s = np.sqrt(np.sum(X[:, n] ** 2))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'max':
        if v is None:
            v = np.ones(K, dtype=np.float64)
        for n in range(N):
            s = np.max(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'z':
        if v is None:
            v = np.zeros(K, dtype=np.float64)
        for n in range(N):
            mu = np.sum(X[:, n]) / K
            sigma = np.sqrt(np.sum((X[:, n] - mu) ** 2) / (K - 1))
            if sigma > threshold:
                X_norm[:, n] = (X[:, n] - mu) / sigma
            else:
                X_norm[:, n] = v

    return X_norm


fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3_F08_C-major-scale_pause.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav)

N, H = 4096, 512
C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
C = C / C.max()

figsize = (6, 2.5)
libfmp.b.plot_chromagram(C, Fs=Fs/H, figsize=figsize, title='Original chromagram')

threshold = 0.000001
C_norm = normalize_feature_sequence(C, norm='2', threshold=threshold)
libfmp.b.plot_chromagram(C_norm, Fs=Fs/H, figsize=figsize,
        title=r'Normalized chromgram ($\ell^2$-norm, $\varepsilon=%f$)' % threshold)

threshold = 0.01
C_norm = normalize_feature_sequence(C, norm='2', threshold=threshold)
libfmp.b.plot_chromagram(C_norm, Fs=Fs/H, figsize=figsize,
        title=r'Normalized chromgram ($\ell^2$-norm, $\varepsilon=%0.2f$)' % threshold)

threshold = 0.01
C_norm = normalize_feature_sequence(C, norm='1', threshold=threshold)
libfmp.b.plot_chromagram(C_norm, Fs=Fs/H, figsize=figsize,
        title=r'Normalized chromgram ($\ell^1$-norm, $\varepsilon=%0.2f$)' % threshold)

threshold = 0.01
C_norm = normalize_feature_sequence(C, norm='max', threshold=threshold)
libfmp.b.plot_chromagram(C_norm, Fs=Fs/H, figsize=figsize,
        title=r'Normalized chromgram (maximum norm, $\varepsilon=%0.2f$)' % threshold)

threshold = 0.01
v = np.zeros(C.shape[0])
C_norm = normalize_feature_sequence(C, norm='max', threshold=threshold, v=v)
libfmp.b.plot_chromagram(C_norm, Fs=Fs/H, figsize=figsize,
        title=r'Normalized chromgram (maximum norm, $v$ is zero vector)')

# %% [markdown]
# ## Normalization by Mean and Variance
#
# Based on mean and variance statistics, one can normalize a feature vector by
# considering its **standard score** z(x) = (x - mu(x)) / sigma(x).

# %%
@jit(nopython=True)
def normalize_feature_sequence_z(X, threshold=0.0001, v=None):
    K, N = X.shape
    X_norm = np.zeros((K, N))

    if v is None:
        v = np.zeros(K)

    for n in range(N):
        mu = np.sum(X[:, n]) / K
        sigma = np.sqrt(np.sum((X[:, n] - mu) ** 2) / (K - 1))
        if sigma > threshold:
            X_norm[:, n] = (X[:, n] - mu) / sigma
        else:
            X_norm[:, n] = v

    return X_norm


threshold = 0.0000001
C_norm = normalize_feature_sequence_z(C, threshold=threshold)
m = np.max(np.abs(C_norm))
libfmp.b.plot_chromagram(C_norm, Fs=Fs/H, figsize=figsize, cmap='seismic', clim=[-m, m],
        title=r'Normalized chromgram (standard score, $\varepsilon=%0.7f$)' % threshold)

threshold = 0.01
C_norm = normalize_feature_sequence_z(C, threshold=threshold)
m = np.max(np.abs(C_norm))
libfmp.b.plot_chromagram(C_norm, Fs=Fs/H, figsize=figsize, cmap='seismic', clim=[-m, m],
        title=r'Normalized chromgram (standard score, $\varepsilon=%0.2f$)' % threshold)

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

# %% [markdown]
# # Signal Reconstruction
#
# Following Section 8.1.2 of [Müller, FMP, Springer 2015], we cover in this notebook the
# important problem of reconstructing a discrete-time signal from a modified STFT. For further
# details, we refer to the classical article by Griffin and Lim.

# %% [markdown]
# ## Introduction
#
# In our procedure for harmonic–percussive separation, the first step was to convert the music
# signal into a time–frequency representation using an STFT. We then manipulated the
# time–frequency representation by applying suitable masking techniques, which resulted in a
# **modified STFT** (MSTFT). Finally, we converted the modified STFT back to the time domain
# by applying an inverse STFT. Even though such an approach appears to be straightforward, it
# turns out that the reconstruction of a time-domain signal from a modified STFT representation
# involves some unanticipated pitfalls. One important question is whether there is a time-domain
# signal whose STFT coincides with the specified MSTFT. In this case, we say that the MSTFT is
# **valid**. In practice, however, it turns out that most of the modified STFTs are not valid.
# In the following, we describe how one typically reconstruct signals from modified STFTs in
# practice and discuss the shortcomings of this procedure.

# %% [markdown]
# ## Notation and Problem Formulation
#
# In the following, we use the same notation as in the FMP notebook on the inverse STFT. Let
# $x:\mathbb{Z}\to\mathbb{R}$ be a discrete-time signal and $\mathcal{X}$ the STFT based on a
# window function $w:[0:N-1]\to\mathbb{R}$ of length $N\in\mathbb{N}$ and a hopsize parameter
# $H\in\mathbb{N}$. Recall that the original signal can be perfectly reconstructed from
# $\mathcal{X}$ in case that the condition
#
# $$\sum_{n\in\mathbb{Z}} w(r-nH)\not= 0$$
#
# holds. Assume that $\mathcal{X}^\mathrm{mod}$ is the given MSTFT. From the reconstruction in
# the modified case, it seems straightforward to apply the following procedure. In a first
# step, we apply the inverse DFT to each of the columns of $\mathcal{X}^\mathrm{mod}$, yielding
#
# $$(v_n(0),\ldots, v_n(N-1))^\top := \mathrm{DFT}_N^{-1} \Big((\mathcal{X}^\mathrm{mod}(n,0),\ldots, \mathcal{X}^\mathrm{mod}d(n,N-1))^\top\Big)$$
#
# for $n\in\mathbb{Z}$. Furthermore, we set $v_n(r):=0$ for $r\in\mathbb{Z}\setminus[0:N-1]$.
# Then, applying an overlap–add technique, we define a signal $x^\mathrm{Rec}:\mathbb{Z}\to\mathbb{R}$
# by setting
#
# $$x^\mathrm{Rec}(r) := \frac{\sum_{n\in\mathbb{Z}} v_n(r-nH)}{\sum_{n\in\mathbb{Z}} w(r-nH)}$$
#
# for $r\in\mathbb{Z}$. Is there something wrong with the signal $x^\mathrm{Rec}$? Yes, there
# is! In general, the STFT $\mathcal{X}^\mathrm{Rec}$ of the signal $x^\mathrm{Rec}$ is not the
# same as the modified STFT $\mathcal{X}^\mathrm{mod}$. The reason is that, when applying the
# windowing to $x^\mathrm{Rec}$, the resulting windowed sections $x^\mathrm{Rec}_n$ usually do
# not agree with the $v_n$.

# %% [markdown]
# ## Example
#
# The STFT $\mathcal{X}^\mathrm{Rec}$ does not match the MSTFT $\mathcal{X}^\mathrm{Mod}$. This
# is due to the fact that the time-shifted analysis windows used for computing the STFT overlap
# with their adjacent windows. For example, computing the second frame of $\mathcal{X}^\mathrm{Rec}$
# also includes information from the first and third windows. Intuitively speaking, by using
# the **overlap–add procedure** in the reconstruction, the information from the previous and
# subsequent frames is reintroduced into the current frame. Note that, even though the signals
# $v_n$ and $x^\mathrm{Rec}_n$ may be different, the respective sums over these signals yield
# the same signal $x^\mathrm{Rec}$.

# %%
import os, sys
import numpy as np
import scipy.signal
import librosa
from matplotlib import pyplot as plt
sys.path.append('..')
import libfmp.b

%matplotlib inline

# Signal
L = 64
t = np.arange(L)/L
omega = 8
x = np.sin(2 * np.pi * omega * t)
x[31] = +1.5
x[32] = -1.5
N = 32
H = N//2
w_type = 'hann'
w = scipy.signal.get_window(w_type, N)

X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann', pad_mode='constant', center=False)
plt.figure(figsize=(9,2.5))
ax = plt.subplot(1,3,1)
libfmp.b.plot_matrix(np.abs(X),ax=[ax],
                xlabel='Time (samples)', ylabel='Frequency (bins)',
                title=r'STFT $\mathcal{X}$')

X_mod = X
X_mod[:,1]=0
ax = plt.subplot(1,3,2)
libfmp.b.plot_matrix(np.abs(X_mod),ax=[ax],
                xlabel='Time (frames)', ylabel='Frequency (bins)',
                title=r'MSTFT $\mathcal{X}^\mathrm{Mod}$')

x_rec = librosa.istft(X_mod, hop_length=H, win_length=N, window='hann', center=False, length=L)
X_rec = librosa.stft(x_rec, n_fft=N, hop_length=H, win_length=N, window='hann', pad_mode='constant', center=False)
ax = plt.subplot(1,3,3)
libfmp.b.plot_matrix(np.abs(X_rec),ax=[ax],
                xlabel='Time (frames)', ylabel='Frequency (bins)',
                title=r'STFT $\mathcal{X}^\mathrm{Rec}$')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,2))
plt.subplot(1,2,1)
plt.plot(x, 'k')
plt.xlim([0,L-1])
plt.ylim([-1.6, 1.6])
plt.title(r'Original signal $x$')
plt.xlabel('Time (samples)')

plt.subplot(1,2,2)
plt.plot(x_rec, 'k')
plt.xlim([0,L-1])
plt.ylim([-1.6, 1.6])
plt.title(r'Reconstructed signal $x^\mathrm{Rec}$ from $\mathcal{X}^\mathrm{Mod}$')
plt.xlabel('Time (samples)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Reconstruction as Optimization Problem
#
# Our example illustrated that the STFT of the reconstructed signal $x^\mathrm{Rec}$ may not
# coincide with the specified MSTFT. More generally, one can show that, regardless of the
# reconstruction method, there may not exist a time-domain signal whose STFT matches a given
# MSTFT. Therefore, an important problem is to find ways for estimating a signal whose STFT
# is at least as close as possible to the MSTFT with regard to a suitably defined distance
# measure. This is exactly the problem that was tackled by Griffin and Lim in their seminal
# paper on **Signal Estimation from Modified Short-Time Fourier Transform**. Following this
# article, we now outline one possible procedure. To measure the distance between a given
# MSTFT $\mathcal{X}^\mathrm{Mod}$ and an STFT $\mathcal{X}'$ of a signal $x'$, we introduce
# the **mean square error** $\Delta(\mathcal{X}^\mathrm{Mod},\mathcal{X}')$ defined by
#
# $$\Delta(\mathcal{X}^\mathrm{Mod},\mathcal{X}'):=\sum_{n\in\mathbb{Z}}\,\,\sum_{k\in[0:N-1]}
#    |\mathcal{X}^\mathrm{Mod}(n,k)-\mathcal{X}'(n,k)|^2.$$
#
# The objective is to find the signal $x^\ast$ whose STFT $\mathcal{X}^\ast$ minimizes this
# error over all possible signals $x'$:
#
# $$x^\ast := \underset{x'}{\mathrm{argmin}} \Delta(\mathcal{X}^\mathrm{Mod},\mathcal{X}').$$
#
# Griffin and Lim showed that this optimization problem has an explicit solution given by
#
# $$x^\ast(r) = \frac{\sum_{n\in\mathbb{Z}} w(r-nH)v_n(r-nH)}{\sum_{n\in\mathbb{Z}} w(r-nH)^2},$$
#
# where the signals $v_n$ are defined as at the beginning of this notebook. Note that this
# procedure is similar in nature to the previously described overlap–add techniques. The major
# difference is that, in the optimal solution, the signals $v_n$ are windowed with the analysis
# window before being overlaid and added. Furthermore, the additional windowing is compensated
# by normalizing with the sum of the squared windows.

# %% [markdown]
# ## Further Notes
#
# * The function `librosa.istft` for computing the inverse STFT uses the optimal overlap–add
#   strategy with the additional windowing. Note that when starting from a valid (unmodified)
#   STFT, both overlap–add strategies yield the same solution with the mean square error being
#   zero.
#
# * Note that in the optimization, the mean square error is measured on the basis of the
#   **complex-valued Fourier coefficients**. In their article, Griffin and Lim propose a second
#   procedure that only considers the mean square error of the **magnitudes** of the MSTFT and
#   the estimated STFT. For this problem, no closed-form solution exists. Instead an iterative
#   optimization procedure is described—an approach that is often referred to as **Griffin–Lim
#   Algorithm**.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

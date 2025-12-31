# %% [markdown]
# # Discrete Fourier Transform (DFT)
#
# In this notebook, we introduce the discrete Fourier transform (DFT) and its
# basic properties. We then study the fast Fourier transform (FFT), which is an
# efficient algorithm to evaluate the DFT.

# %% [markdown]
# ## Inner Product
#
# An important concept for understanding the Fourier transform is the **inner product**
# for a complex vector space C^N. Given two complex vectors x, y in C^N, the inner
# product between x and y is defined as:
#
# <x | y> := sum_{n=0}^{N-1} x(n) * conj(y(n))
#
# The absolute value of the inner product may be interpreted as a measure of similarity
# between x and y.
#
# Note: When using np.vdot, the complex conjugate is performed on the first argument.
# Therefore, for computing <x | y>, call np.vdot(y, x).

# %%
import numpy as np
from numba import jit

x = np.array([1.0, 1j, 1.0 + 1.0j])
y = np.array([1.1, 1j, 0.9 + 1.1j])
print('Vectors of high similarity:', np.abs(np.vdot(y, x)))

x = np.array([1.0, 1j, 1.0 + 1j])
y = np.array([1.1, -1j, 0.1])
print('Vectors of low similarity:', np.abs(np.vdot(y, x)))

# %% [markdown]
# ## Definition of DFT
#
# Let x in C^N be a vector of length N. The **discrete Fourier transform** (DFT)
# of x is defined by:
#
# X(k) := sum_{n=0}^{N-1} x(n) * exp(-2*pi*i*k*n/N)
#
# for k in [0:N-1]. The DFT can be expressed as inner products:
# X(k) = <x | u_k>
# where u_k(n) := exp(2*pi*i*k*n/N)

# %%
from matplotlib import pyplot as plt
%matplotlib inline

N = 64
n = np.arange(N)
k = 3
x = np.cos(2 * np.pi * (k * n / N) + (1.2 * np.random.rand(N) - 0.0))

plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(n, x, 'k', marker='.', markersize='10', linewidth=2.0, label='$x$')
plt.xlabel('Time (samples)')
k = 3
u_k_real = np.cos(2 * np.pi * k * n / N)
u_k_imag = -np.sin(2 * np.pi * k * n / N)
u_k = u_k_real + u_k_imag * 1j
sim_complex = np.vdot(u_k, x)
sim_abs = np.abs(sim_complex)
plt.title(r'Signal $x$ and some $u_k$ (k=3) having high similarity: Re($X(k)$) = %0.2f, Im($X(k)$) = %0.2f,  $|X(k)|$=%0.2f' % (sim_complex.real, sim_complex.imag, sim_abs))
plt.plot(n, u_k_real, 'r', marker='.', markersize='5', linewidth=1.0, linestyle=':', label=r'$\mathrm{Re}(\overline{\mathbf{u}}_k)$')
plt.plot(n, u_k_imag, 'b', marker='.', markersize='5', linewidth=1.0, linestyle=':', label=r'$\mathrm{Im}(\overline{\mathbf{u}}_k)$')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(n, x, 'k', marker='.', markersize='10', linewidth=2.0, label='$x$')
plt.xlabel('Time (samples)')
k = 5
u_k_real = np.cos(2 * np.pi * k * n / N)
u_k_imag = -np.sin(2 * np.pi * k * n / N)
u_k = u_k_real + u_k_imag * 1j
sim_complex = np.vdot(u_k, x)
sim_abs = np.abs(sim_complex)
plt.title(r'Signal $x$ and some $u_k$ (k=5) having low similarity: Re($X(k)$) = %0.2f, Im($X(k)$) = %0.2f,  $|X(k)|$=%0.2f' % (sim_complex.real, sim_complex.imag, sim_abs))
plt.plot(n, u_k_real, 'r', marker='.', markersize='5', linewidth=1.0, linestyle=':', label=r'$\mathrm{Re}(\overline{\mathbf{u}}_k)$')
plt.plot(n, u_k_imag, 'b', marker='.', markersize='5', linewidth=1.0, linestyle=':', label=r'$\mathrm{Im}(\overline{\mathbf{u}}_k)$')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## DFT Matrix
#
# Being a linear operator C^N -> C^N, the DFT can be expressed by an N x N matrix.
# The DFT matrix DFT_N is given by:
# DFT_N(n, k) = exp(-2*pi*i*k*n/N)

# %%
@jit(nopython=True)
def generate_matrix_dft(N, K):
    """Generates a DFT (discrete Fourier transform) matrix

    Args:
        N (int): Number of samples
        K (int): Number of frequency bins

    Returns:
        dft (np.ndarray): The DFT matrix
    """
    dft = np.zeros((K, N), dtype=np.complex128)
    for n in range(N):
        for k in range(K):
            dft[k, n] = np.exp(-2j * np.pi * k * n / N)
    return dft


N = 32
dft_mat = generate_matrix_dft(N, N)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.title('$\mathrm{Re}(\mathrm{DFT}_N)$')
plt.imshow(np.real(dft_mat), origin='lower', cmap='seismic', aspect='equal')
plt.xlabel('Time index $n$')
plt.ylabel('Frequency index $k$')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('$\mathrm{Im}(\mathrm{DFT}_N)$')
plt.imshow(np.imag(dft_mat), origin='lower', cmap='seismic', aspect='equal')
plt.xlabel('Time index $n$')
plt.ylabel('Frequency index $k$')
plt.colorbar()
plt.tight_layout()
plt.show()

# %%
@jit(nopython=True)
def dft(x):
    """Compute the discrete Fourier transform (DFT)

    Args:
        x (np.ndarray): Signal to be transformed

    Returns:
        X (np.ndarray): Fourier transform of x
    """
    x = x.astype(np.complex128)
    N = len(x)
    dft_mat = generate_matrix_dft(N, N)
    return np.dot(dft_mat, x)


N = 128
n = np.arange(N)
k = 10
x = np.cos(2 * np.pi * (k * n / N) + 2 * (np.random.rand(N) - 0.5))
X = dft(x)

plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.title('$x$')
plt.plot(x, 'k')
plt.xlabel('Time (index $n$)')

plt.subplot(1, 2, 2)
plt.title('$|X|$')
plt.plot(np.abs(X), 'k')
plt.xlabel('Frequency (index $k$)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Fast Fourier Transform (FFT)
#
# The FFT is a fast algorithm to compute the DFT. It was originally found by Gauss
# in about 1805 and then rediscovered by Cooley and Tukey in 1965. The FFT reduces
# the overall number of operations from O(N^2) to O(N*log2(N)).
#
# The FFT algorithm is based on the observation that applying a DFT of even size N=2M
# can be expressed in terms of applying two DFTs of half the size M.

# %%
@jit(nopython=True)
def twiddle(N):
    """Generate the twiddle factors used in FFT

    Args:
        N (int): Number of samples

    Returns:
        sigma (np.ndarray): The twiddle factors
    """
    k = np.arange(N // 2)
    sigma = np.exp(-2j * np.pi * k / N)
    return sigma


@jit(nopython=True)
def fft(x):
    """Compute the fast Fourier transform (FFT)

    Args:
        x (np.ndarray): Signal to be transformed

    Returns:
        X (np.ndarray): Fourier transform of x
    """
    x = x.astype(np.complex128)
    N = len(x)
    log2N = np.log2(N)
    assert log2N == int(log2N), 'N must be a power of two!'
    X = np.zeros(N, dtype=np.complex128)

    if N == 1:
        return x
    else:
        this_range = np.arange(N)
        A = fft(x[this_range % 2 == 0])
        B = fft(x[this_range % 2 == 1])
        C = twiddle(N) * B
        X[:N // 2] = A + C
        X[N // 2:] = A - C
        return X


N = 16
n = np.arange(N)
k = 4
x = np.cos(2 * np.pi * (k * n / N) + 2 * (np.random.rand(N) - 0.5))
X_via_dft = dft(x)
X_via_fft = fft(x)

plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.title('$x$')
plt.plot(x, 'k', marker='.', markersize=12)
plt.xlabel('Time (index $n$)')

plt.subplot(1, 2, 2)
plt.title('$|X|$')
plt.plot(np.abs(X_via_dft), 'k', marker='.', markersize=18, label='dft')
plt.plot(np.abs(X_via_fft), linestyle='--', color='pink', marker='.', markersize=6, label='fft')
plt.xlabel('Frequency (index $k$)')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Computational Complexity
#
# The FFT reduces the overall number of operations from O(N^2) to O(N*log2(N)).
# For example, using N=2^10=1024, the FFT requires roughly N*log2(N)=10240 instead
# of N^2=1048576 operations in the naive approach.

# %%
N = 512
n = np.arange(N)
x = np.sin(2 * np.pi * 5 * n / N)

print('Timing for DFT: ', end='')
%timeit dft(x)
print('Timing for FFT: ', end='')
%timeit fft(x)

# %%
import timeit

Ns = [2 ** n for n in range(5, 11)]
times_dft = []
times_fft = []
execuctions = 5

for N in Ns:
    n = np.arange(N)
    x = np.sin(2 * np.pi * 5 * n / N)

    time_dft = timeit.timeit(lambda: dft(x), number=execuctions) / execuctions
    time_fft = timeit.timeit(lambda: fft(x), number=execuctions) / execuctions
    times_dft.append(time_dft)
    times_fft.append(time_fft)

plt.figure(figsize=(10, 4))

plt.plot(Ns, times_dft, '-xk', label='DFT')
plt.plot(Ns, times_fft, '-xr', label='FFT')
plt.xticks(Ns)
plt.legend()
plt.grid()
plt.xlabel('$N$')
plt.ylabel('Runtime (seconds)')
plt.show()

# %% [markdown]
# ## Further Notes
#
# The dft and fft functions have been included into libfmp.

# %%
import sys
sys.path.append('..')
import libfmp.c2

N = 16
n = np.arange(N)
k = 4
x = np.cos(2 * np.pi * (k * n / N) + 2 * (np.random.rand(N) - 0.5))
X_via_dft = libfmp.c2.dft(x)
X_via_fft = libfmp.c2.fft(x)

plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.title('$x$')
plt.plot(x, 'k', marker='.', markersize=12)
plt.xlabel('Time (index $n$)')

plt.subplot(1, 2, 2)
plt.title('$|X|$')
plt.plot(np.abs(X_via_dft), 'k', marker='.', markersize=18, label='dft')
plt.plot(np.abs(X_via_fft), linestyle='--', color='pink', marker='.', markersize=6, label='fft')
plt.xlabel('Frequency (index $k$)')
plt.legend()

plt.tight_layout()
plt.show()

N = 512
n = np.arange(N)
x = np.sin(2 * np.pi * 5 * n / N)

print('Timing for DFT: ', end='')
%timeit libfmp.c2.dft(x)
print('Timing for FFT: ', end='')
%timeit libfmp.c2.fft(x)

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Frank Zalkow and Meinard Muller.

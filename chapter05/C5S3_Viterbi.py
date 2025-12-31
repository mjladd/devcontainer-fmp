# %% [markdown]
# # Viterbi Algorithm
#
# Following Section 5.3.3.2 of [Müller, FMP, Springer 2015], we describe in this notebook the
# Viterbi algorithm, which efficiently solves the uncovering problem of hidden Markov models
# (HMMs). For a detailed introduction of HMMs, we refer to the famous tutorial paper by Rabiner.

# %% [markdown]
# ## Uncovering Problem and Viterbi Algorithm
#
# We now consider the uncovering problem in more detail. The goal is to find the single state
# sequence $S=(s_{1},s_{2},\ldots,s_{N})$ that "best explains" a given observation sequence
# $O=(o_{1},o_{2},\ldots,o_{N})$. In the chord recognition scenario, the observation sequence
# is a sequence of chroma vectors extracted from an audio recording. The optimal state sequence
# is then the sequence of chord labels.
#
# As optimization criterion, we choose the state sequence $S^\ast$ that yields the highest
# probability when evaluated against the observation sequence $O$:
#
# $$\mathrm{Prob}^\ast = \underset{S=(s_{1},s_{2},\ldots,s_{N})}{\max} \,\,P[O,S|\Theta]$$
#
# $$S^\ast = \underset{S=(s_{1},s_{2},\ldots,s_{N})}{\mathrm{argmax}} \,\,\,P[O,S|\Theta]$$
#
# The Viterbi algorithm is based on dynamic programming and recursively computes an optimal
# state sequence. We define:
#
# $$\mathbf{D}(i,n):=\underset{(s_1,\ldots,s_n)}{\max} P[O(1:n),(s_1,\ldots, s_{n-1},s_n=\alpha_i)|\Theta]$$
#
# The computational complexity of the Viterbi algorithm is $O(N\cdot I^2)$, which is much
# better than $O(I^N)$ required for the naive approach.

# %% [markdown]
# ## Toy Example
#
# We illustrate the principle of the Viterbi algorithm using a toy example with three states
# corresponding to the major chords C, G, and F.

# %% [markdown]
# ## Implementation of Viterbi Algorithm
#
# We provide an implementation of the Viterbi algorithm. Note that due to Python conventions,
# the indexing starts with index 0.

# %%
import numpy as np
from numba import jit

@jit(nopython=True)
def viterbi(A, C, B, O):
    """Viterbi algorithm for solving the uncovering problem

    Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        A (np.ndarray): State transition probability matrix of dimension I x I
        C (np.ndarray): Initial state distribution  of dimension I
        B (np.ndarray): Output probability matrix of dimension I x K
        O (np.ndarray): Observation sequence of length N

    Returns:
        S_opt (np.ndarray): Optimal state sequence of length N
        D (np.ndarray): Accumulated probability matrix
        E (np.ndarray): Backtracking matrix
    """
    I = A.shape[0]    # Number of states
    N = len(O)  # Length of observation sequence

    # Initialize D and E matrices
    D = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D[:, 0] = np.multiply(C, B[:, O[0]])

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_product = np.multiply(A[:, i], D[:, n-1])
            D[i, n] = np.max(temp_product) * B[i, O[n]]
            E[i, n-1] = np.argmax(temp_product)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D[:, -1])
    for n in range(N-2, -1, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    return S_opt, D, E

# Define model parameters
A = np.array([[0.8, 0.1, 0.1],
              [0.2, 0.7, 0.1],
              [0.1, 0.3, 0.6]])

C = np.array([0.6, 0.2, 0.2])

B = np.array([[0.7, 0.0, 0.3],
              [0.1, 0.9, 0.0],
              [0.0, 0.2, 0.8]])


O = np.array([0, 2, 0, 2, 2, 1]).astype(np.int32)
#O = np.array([1]).astype(np.int32)
#O = np.array([1, 2, 0, 2, 2, 1]).astype(np.int32)

# Apply Viterbi algorithm
S_opt, D, E = viterbi(A, C, B, O)
#
print('Observation sequence:   O = ', O)
print('Optimal state sequence: S = ', S_opt)
np.set_printoptions(formatter={'float': "{: 7.4f}".format})
print('D =', D, sep='\n')
np.set_printoptions(formatter={'float': "{: 7.0f}".format})
print('E =', E, sep='\n')

# %% [markdown]
# ## Log-Domain Implementation of Viterbi Algorithm
#
# In each iteration of the Viterbi algorithm, the accumulated probability values are multiplied
# with two probability values from $A$ and $B$. Since all probability values lie in the interval
# $[0,1]$, the product of such values decreases exponentially with the number of iterations.
# For input sequences with large $N$, the values typically become extremely small, which may
# lead to numerical underflow.
#
# A well-known trick when dealing with products of probability values is to work in the
# log-domain. We apply a logarithm to all probability values and replace multiplication by
# summation. Since the logarithm is a strictly monotonous function, ordering relations are
# preserved in the log-domain.

# %%
@jit(nopython=True)
def viterbi_log(A, C, B, O):
    """Viterbi algorithm (log variant) for solving the uncovering problem

    Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        A (np.ndarray): State transition probability matrix of dimension I x I
        C (np.ndarray): Initial state distribution  of dimension I
        B (np.ndarray): Output probability matrix of dimension I x K
        O (np.ndarray): Observation sequence of length N

    Returns:
        S_opt (np.ndarray): Optimal state sequence of length N
        D_log (np.ndarray): Accumulated log probability matrix
        E (np.ndarray): Backtracking matrix
    """
    I = A.shape[0]    # Number of states
    N = len(O)  # Length of observation sequence
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_log = np.log(B + tiny)

    # Initialize D and E matrices
    D_log = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D_log[:, 0] = C_log + B_log[:, O[0]]

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + D_log[:, n-1]
            D_log[i, n] = np.max(temp_sum) + B_log[i, O[n]]
            E[i, n-1] = np.argmax(temp_sum)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N-2, -1, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    return S_opt, D_log, E

# Apply Viterbi algorithm (log variant)
S_opt, D_log, E = viterbi_log(A, C, B, O)

print('Observation sequence:   O = ', O)
print('Optimal state sequence: S = ', S_opt)
np.set_printoptions(formatter={'float': "{: 7.2f}".format})
print('D_log =', D_log, sep='\n')
np.set_printoptions(formatter={'float': "{: 7.4f}".format})
print('exp(D_log) =', np.exp(D_log), sep='\n')
np.set_printoptions(formatter={'float': "{: 7.0f}".format})
print('E =', E, sep='\n')

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Christof Weiß.

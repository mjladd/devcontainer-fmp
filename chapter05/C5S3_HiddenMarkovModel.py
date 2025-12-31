# %% [markdown]
# # Hidden Markov Model (HMM)
#
# Motivated by the chord recognition problem, we give in this notebook an overview of hidden
# Markov models (HMMs) and introduce three famous algorithmic problems related with HMMs
# following Section 5.3 of [Müller, FMP, Springer 2015]. For a detailed introduction of HMMs,
# we refer to the famous tutorial paper by Rabiner.

# %% [markdown]
# ## Markov Chains
#
# Certain transitions from one chord to another are more likely than others. To capture such
# likelihoods, one can employ a concept called Markov chains. Abstracting from our chord
# recognition scenario, we assume that the chord types to be considered are represented by
# a set:
#
# $$\mathcal{A}:=\{\alpha_{1},\alpha_{2},\ldots,\alpha_{I}\}$$
#
# of size $I\in\mathbb{N}$. The elements $\alpha_{i}$ for $i\in[1:I]$ are referred to as states.
# The change from one state to another is specified according to a set of probabilities.
# The Markov property states that the probability of a change from the current state to the
# next state only depends on the current state:
#
# $$P[s_{n+1}=\alpha_{j}|s_{n}=\alpha_{i},s_{n-1}=\alpha_{k},\ldots] = P[s_{n+1}=\alpha_{j}|s_{n}=\alpha_{i}]$$
#
# The state transition probabilities are given by:
#
# $$a_{ij} := P[s_{n+1}=\alpha_{j} | s_{n}=\alpha_{i}] \in [0,1]$$
#
# These coefficients obey the stochastic constraint $\sum_{j=1}^{I} a_{ij} = 1$ and can be
# expressed by an $(I\times I)$ matrix $A$.

# %% [markdown]
# ## Hidden Markov Models
#
# In our chord recognition scenario, rather than observing a sequence of chord types, we
# observe a sequence of chroma vectors that are somehow related to the chord types. This
# leads to an extension of Markov chains to a statistical model referred to as a hidden
# Markov model (HMM).
#
# For discrete HMMs, the output space is a finite set:
#
# $$\mathcal{B} = \{\beta_{1},\beta_{2},\ldots,\beta_{K}\}$$
#
# of size $K\in\mathbb{N}$ consisting of distinct observation symbols. The emission probabilities
# are specified by coefficients $b_{ik}\in[0,1]$ for $i\in[1:I]$ and $k\in[1:K]$.
#
# In summary, an HMM is specified by a tuple:
#
# $$\Theta:=(\mathcal{A},A,C,\mathcal{B},B)$$

# %% [markdown]
# ## Example HMM
#
# We define the state transition probability matrix $A$ and the output probability $B$ for
# a simple example with three states corresponding to the major chords C, G, and F.

# %%
import numpy as np
from sklearn.preprocessing import normalize

A = np.array([[0.8, 0.1, 0.1],
              [0.2, 0.7, 0.1],
              [0.1, 0.3, 0.6]])

C = np.array([0.6, 0.2, 0.2])

B = np.array([[0.7, 0.0, 0.3],
              [0.1, 0.9, 0.0],
              [0.0, 0.2, 0.8]])

# %% [markdown]
# ## HMM-Based Sequence Generation
#
# Once an HMM is specified by $\Theta:=(\mathcal{A},A,C,\mathcal{B},B)$, it can be used for
# various analysis and synthesis applications. The generation procedure is as follows:
#
# 1. Set $n=1$ and choose an initial state $s_n=\alpha_i$ according to the initial state
#    distribution $C$.
# 2. Generate an observation $o_n=\beta_k$ according to the emission probability in state
#    $s_n=\alpha_i$ (specified by the $i^{\mathrm{th}}$ row of $B$).
# 3. If $n=N$ then terminate. Otherwise, transit to the new state $s_{n+1}=\alpha_{j}$
#    according to the state transition probability at state $s_n=\alpha_i$. Then increase
#    $n$ by one and return to step 2.

# %%
def generate_sequence_hmm(N, A, C, B, details=False):
    """Generate observation and state sequence from given HMM

    Notebook: C5/C5S3_HiddenMarkovModel.ipynb

    Args:
        N (int): Number of observations to be generated
        A (np.ndarray): State transition probability matrix of dimension I x I
        C (np.ndarray): Initial state distribution of dimension I
        B (np.ndarray): Output probability matrix of dimension I x K
        details (bool): If "True" then shows details (Default value = False)

    Returns:
        O (np.ndarray): Observation sequence of length N
        S (np.ndarray): State sequence of length N
    """
    assert N > 0, "N should be at least one"
    I = A.shape[1]
    K = B.shape[1]
    assert I == A.shape[0], "A should be an I-square matrix"
    assert I == C.shape[0], "Dimension of C should be I"
    assert I == B.shape[0], "Column-dimension of B should be I"

    O = np.zeros(N, int)
    S = np.zeros(N, int)
    for n in range(N):
        if n == 0:
            i = np.random.choice(np.arange(I), p=C)
        else:
            i = np.random.choice(np.arange(I), p=A[i, :])
        k = np.random.choice(np.arange(K), p=B[i, :])
        S[n] = i
        O[n] = k
        if details:
            print('n = %d, S[%d] = %d, O[%d] = %d' % (n, n, S[n], n, O[n]))
    return O, S

N = 10
O, S = generate_sequence_hmm(N, A, C, B, details=True)
print('State sequence S:      ', S)
print('Observation sequence O:', O)

# %% [markdown]
# ## Parameter Estimation from Sequences
#
# As a sanity check for the plausibility of our sequence generation approach, we estimate
# the original transition probability matrix $A$ and the output probability matrix $B$ from
# a generated observation sequence $O$ and state sequence $S$.
#
# * To obtain an estimate of $a_{ij}$, we count all transitions from $n$ to $n+1$ with
#   $S(n)=\alpha_i$ and $S(n+1)=\alpha_j$ and divide by the total number of transitions
#   starting with $\alpha_i$.
# * To obtain an estimate of $b_{ik}$, we count the number of occurrences $n$ with
#   $S(n)=\alpha_i$ and $O(n)=\beta_k$ and divide by the total number of occurrences of
#   $\alpha_i$ in $S$.

# %%
def estimate_hmm_from_o_s(O, S, I, K):
    """Estimate the state transition and output probability matrices from
    a given observation and state sequence

    Notebook: C5/C5S3_HiddenMarkovModel.ipynb

    Args:
        O (np.ndarray): Observation sequence of length N
        S (np.ndarray): State sequence of length N
        I (int): Number of states
        K (int): Number of observation symbols

    Returns:
        A_est (np.ndarray): State transition probability matrix of dimension I x I
        B_est (np.ndarray): Output probability matrix of dimension I x K
    """
    # Estimate A
    A_est = np.zeros([I, I])
    N = len(S)
    for n in range(N-1):
        i = S[n]
        j = S[n+1]
        A_est[i, j] += 1
    A_est = normalize(A_est, axis=1, norm='l1')

    # Estimate B
    B_est = np.zeros([I, K])
    for i in range(I):
        for k in range(K):
            B_est[i, k] = np.sum(np.logical_and(S == i, O == k))
    B_est = normalize(B_est, axis=1, norm='l1')
    return A_est, B_est

N = 100
print('======== Estimation results when using N = %d ========' % N)
O, S = generate_sequence_hmm(N, A, C, B, details=False)
A_est, B_est = estimate_hmm_from_o_s(O, S, A.shape[1], B.shape[1])
np.set_printoptions(formatter={'float': "{: 7.3f}".format})
print('A =', A, sep='\n')
print('A_est =', A_est, sep='\n')
print('B =', B, sep='\n')
print('B_est =', B_est, sep='\n')

N = 10000
print('======== Estimation results when using N = %d ========' % N)
O, S = generate_sequence_hmm(N, A, C, B, details=False)
A_est, B_est = estimate_hmm_from_o_s(O, S, A.shape[1], B.shape[1])
np.set_printoptions(formatter={'float': "{: 7.3f}".format})
print('A =', A, sep='\n')
print('A_est =', A_est, sep='\n')
print('B =', B, sep='\n')
print('B_est =', B_est, sep='\n')

# %% [markdown]
# ## Three Problems for HMMs
#
# We will now look at three famous algorithmic problems for HMMs that concern the
# specification of the free model parameters and the evaluation of observation sequences.
#
# ### 1. Evaluation Problem
#
# Given an HMM specified by $\Theta=(\mathcal{A},A,C,\mathcal{B},B)$ and an observation
# sequence $O=(o_{1},o_{2},\ldots,o_{N})$, the task is to compute the probability
# $P[O|\Theta]$ of the observation sequence given the model.
#
# The overall probability can be computed as:
#
# $$P[O|\Theta] = \sum_{S: |S|=N}P[O,S|\Theta]$$
#
# This leads to $I^N$ summands, a number that is exponential in the length $N$. The
# Forward-Backward Algorithm provides a more efficient solution requiring $O(I^2N)$ operations.
#
# ### 2. Uncovering Problem
#
# The goal is to find the single state sequence $S=(s_{1},s_{2},\ldots,s_{N})$ that "best
# explains" the observation sequence. The Viterbi algorithm efficiently solves this problem.
#
# ### 3. Estimation Problem
#
# Given an observation sequence $O$, the objective is to determine the free model parameters
# that maximize the probability $P[O|\Theta]$. The Baum-Welch Algorithm provides an iterative
# procedure for finding locally optimal solutions.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

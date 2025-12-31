# %% [markdown]
# # Digital Signals: Quantization
#
# Sampling transforms a continuous time axis into a discrete time axis.
# **Quantization** replaces the continuous range of possible amplitudes by a
# discrete range of possible values.

# %% [markdown]
# ## Uniform Quantization
#
# A quantization can be modeled by a function Q: R -> Gamma, called the **quantizer**,
# which assigns to each amplitude value a in R a value Q(a) in Gamma.
#
# A typical **uniform** quantizer with quantization step size Delta spaces the
# quantization levels uniformly:
#
# Q(a) := sign(a) * Delta * floor(|a|/Delta + 1/2)
#
# For Delta = 1, this is simple rounding to the nearest integer.

# %% [markdown]
# ## Quantization Error
#
# Quantization is generally a lossy operation. The difference between the actual
# value and the quantized value is called the **quantization error**.
#
# Reducing the quantization step size typically leads to smaller errors, but
# increases the number of bits needed to encode the values.

# %%
import numpy as np
from matplotlib import pyplot as plt
import os
import librosa
from IPython.display import Audio, display

import sys
sys.path.append('..')
import libfmp.c1
%matplotlib inline


def quantize_uniform(x, quant_min=-1.0, quant_max=1.0, quant_level=5):
    """Uniform quantization approach

    Args:
        x (np.ndarray): Original signal
        quant_min (float): Minimum quantization level
        quant_max (float): Maximum quantization level
        quant_level (int): Number of quantization levels

    Returns:
        x_quant (np.ndarray): Quantized signal
    """
    x_normalize = (x - quant_min) * (quant_level - 1) / (quant_max - quant_min)
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.around(x_normalize)
    x_quant = (x_normalize_quant) * (quant_max - quant_min) / (quant_level - 1) + quant_min
    return x_quant


def plot_graph_quant_function(ax, quant_min=-1.0, quant_max=1.0, quant_level=256,
                               mu=255.0, quant='uniform'):
    """Helper function for plotting quantization function and error"""
    x = np.linspace(quant_min, quant_max, 1000)
    if quant == 'uniform':
        x_quant = quantize_uniform(x, quant_min=quant_min, quant_max=quant_max,
                                   quant_level=quant_level)
        quant_stepsize = (quant_max - quant_min) / (quant_level - 1)
        title = r'$\lambda = %d, \Delta=%0.2f$' % (quant_level, quant_stepsize)
    if quant == 'nonuniform':
        x_quant = quantize_nonuniform_mu(x, mu=mu, quant_level=quant_level)
        title = r'$\lambda = %d, \mu=%0.1f$' % (quant_level, mu)
    error = np.abs(x_quant - x)
    ax.plot(x, x, color='k', label='Original amplitude')
    ax.plot(x, x_quant, color='b', label='Quantized amplitude')
    ax.plot(x, error, 'r--', label='Quantization error')
    ax.set_title(title)
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Quantized amplitude/error')
    ax.set_xlim([quant_min, quant_max])
    ax.set_ylim([quant_min, quant_max])
    ax.grid('on')
    ax.legend()


plt.figure(figsize=(12, 4))
ax = plt.subplot(1, 3, 1)
plot_graph_quant_function(ax, quant_min=-1, quant_max=4, quant_level=3)
ax = plt.subplot(1, 3, 2)
plot_graph_quant_function(ax, quant_min=-2, quant_max=2, quant_level=4)
ax = plt.subplot(1, 3, 3)
plot_graph_quant_function(ax, quant_min=-1, quant_max=1, quant_level=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Example: Uniform Quantization of Sinusoid

# %%
def plot_signal_quant(x, t, x_quant, figsize=(8, 2), xlim=None, ylim=None, title=''):
    """Helper function for plotting a signal and its quantized version"""
    plt.figure(figsize=figsize)
    plt.plot(t, x, color='gray', linewidth=1.0, linestyle='-', label='Original signal')
    plt.plot(t, x_quant, color='red', linewidth=2.0, linestyle='-', label='Quantized signal')
    if xlim is None:
        plt.xlim([0, t[-1]])
    else:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend(loc='upper right', framealpha=1)
    plt.tight_layout()
    plt.show()


dur = 5
x, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=1000, amp=1, freq=1, phase=0.0)

quant_min = -1
quant_max = 1
quant_level = 5
x_quant = quantize_uniform(x, quant_min=quant_min, quant_max=quant_max,
                           quant_level=quant_level)
plot_signal_quant(x, t, x_quant, xlim=[0, dur], ylim=[-1.3, 1.3],
                  title=r'Uniform quantization with min=$%0.1f$, max=$%0.1f$, $\lambda$=$%d$' %
                        (quant_min, quant_max, quant_level))

quant_min = -0.5
quant_max = 1
quant_level = 3
x_quant = quantize_uniform(x, quant_min=quant_min, quant_max=quant_max,
                           quant_level=quant_level)
plot_signal_quant(x, t, x_quant, xlim=[0, dur], ylim=[-1.3, 1.3],
                  title=r'Uniform quantization with min=$%0.1f$, max=$%0.1f$, $\lambda$=$%d$' %
                        (quant_min, quant_max, quant_level))

quant_min = -1.2
quant_max = 1.2
quant_level = 4
x_quant = quantize_uniform(x, quant_min=quant_min, quant_max=quant_max,
                           quant_level=quant_level)
plot_signal_quant(x, t, x_quant, xlim=[0, dur], ylim=[-1.3, 1.3],
                  title=r'Uniform quantization with min=$%0.1f$, max=$%0.1f$, $\lambda$=$%d$' %
                        (quant_min, quant_max, quant_level))

# %% [markdown]
# ## Quantization Noise
#
# The distortions introduced by quantization are referred to as **quantization noise**.

# %%
def display_signal_quant(x, Fs, number_of_bits):
    quant_level = 2 ** number_of_bits
    x_quant = quantize_uniform(x, quant_min=-1, quant_max=1, quant_level=quant_level)
    print('Signal after uniform quantization (%d bits):' % number_of_bits, flush=True)
    display(Audio(x_quant, rate=Fs))
    return x_quant


file_name = os.path.join('..', 'data', 'C2', 'FMP_C2_Sampling_C-major-scale.wav')
x, Fs = librosa.load(file_name, sr=11025)

print('Original audio signal (16 bits):', flush=True)
display(Audio(x, rate=Fs))

x_quant = display_signal_quant(x=x, Fs=Fs, number_of_bits=8)
x_quant = display_signal_quant(x=x, Fs=Fs, number_of_bits=4)
x_quant = display_signal_quant(x=x, Fs=Fs, number_of_bits=2)

# %% [markdown]
# ## Nonuniform Quantization
#
# In **nonuniform** quantization, the quantization levels are not equidistant.
# For audio signals, one often uses **logarithmic** spacing because human perception
# of sound intensity is logarithmic.
#
# The **mu-law encoding** is:
# F_mu(v) = sign(v) * ln(1 + mu * |v|) / ln(1 + mu)
#
# The **mu-law decoding** is:
# F_mu^{-1}(v) = sign(v) * ((1 + mu)^|v| - 1) / mu

# %%
def encoding_mu_law(v, mu=255.0):
    """mu-law encoding"""
    v_encode = np.sign(v) * (np.log(1.0 + mu * np.abs(v)) / np.log(1.0 + mu))
    return v_encode


def decoding_mu_law(v, mu=255.0):
    """mu-law decoding"""
    v_decode = np.sign(v) * (1.0 / mu) * ((1.0 + mu) ** np.abs(v) - 1.0)
    return v_decode


def plot_mu_law(mu=255.0, figsize=(8.5, 4)):
    """Helper function for plotting mu-law encoding/decoding"""
    values = np.linspace(-1, 1, 1000)
    values_encoded = encoding_mu_law(values, mu=mu)
    values_decoded = encoding_mu_law(values, mu=mu)

    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 2, 1)
    ax.plot(values, values, color='k', label='Original values')
    ax.plot(values, values_encoded, color='b', label='Encoded values')
    ax.set_title(r'$\mu$-law encoding with $\mu=%.0f$' % mu)
    ax.set_xlabel('$v$')
    ax.set_ylabel(r'$F_\mu(v)$')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.grid('on')
    ax.legend()

    ax = plt.subplot(1, 2, 2)
    ax.plot(values, values, color='k', label='Original values')
    ax.plot(values, values_decoded, color='b', label='Decoded values')
    ax.set_title(r'$\mu$-law decoding with $\mu=%.0f$' % mu)
    ax.set_xlabel('$v$')
    ax.set_ylabel(r'$F_\mu^{-1}(v)$')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.grid('on')
    ax.legend()

    plt.tight_layout()
    plt.show()


plot_mu_law(mu=255.0)
plot_mu_law(mu=7.0)

# %% [markdown]
# ## Implementation of Nonuniform Quantization
#
# First encode using F_mu, then apply uniform quantization, then decode using F_mu^{-1}.

# %%
def quantize_nonuniform_mu(x, mu=255.0, quant_level=256):
    """Nonuniform quantization approach using mu-encoding"""
    x_en = encoding_mu_law(x, mu=mu)
    x_en_quant = quantize_uniform(x_en, quant_min=-1, quant_max=1, quant_level=quant_level)
    x_quant = decoding_mu_law(x_en_quant, mu=mu)
    return x_quant


dur = 5
x, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=1000, amp=1, freq=1, phase=0.0)

quant_level = 8
x_quant = quantize_uniform(x, quant_min=-1, quant_max=1, quant_level=quant_level)
plot_signal_quant(x, t, x_quant, xlim=[0, dur], ylim=[-1.3, 1.3],
                  title=r'Uniform quantization with $\lambda$=$%d$' % (quant_level))

mu = 7
x_quant = quantize_nonuniform_mu(x, mu=mu, quant_level=quant_level)
plot_signal_quant(x, t, x_quant, xlim=[0, dur], ylim=[-1.3, 1.3],
                  title=r'Nonuniform quantization with $\mu$=$%d$ and $\lambda$=$%d$' % (mu, quant_level))

# %%
# Graph of nonuniform quantization function
plt.figure(figsize=(12, 4))
ax = plt.subplot(1, 3, 1)
plot_graph_quant_function(ax, mu=3, quant_level=4, quant='nonuniform')
ax = plt.subplot(1, 3, 2)
plot_graph_quant_function(ax, mu=7, quant_level=8, quant='nonuniform')
ax = plt.subplot(1, 3, 3)
plot_graph_quant_function(ax, mu=15, quant_level=16, quant='nonuniform')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller, Tim Zunner, and Michael Krause.

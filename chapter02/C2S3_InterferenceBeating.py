# %% [markdown]
# # Interference and Beating
#
# We discuss the effect of signal interference and the phenomenon of beating.

# %% [markdown]
# ## Interference
#
# **Interference** occurs when a wave is superimposed with another wave of similar
# frequency.
#
# - **Constructive interference**: Crests meet crests, magnitudes add up
# - **Destructive interference**: Crests meet troughs, magnitudes cancel out

# %%
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('..')
import libfmp.c1
%matplotlib inline


def plot_interference(x1, x2, t, figsize=(8, 2), xlim=None, ylim=None, title=''):
    """Helper function for plotting two signals and their superposition"""
    plt.figure(figsize=figsize)
    plt.plot(t, x1, color='gray', linewidth=1.0, linestyle='-', label='x1')
    plt.plot(t, x2, color='cyan', linewidth=1.0, linestyle='-', label='x2')
    plt.plot(t, x1 + x2, color='red', linewidth=2.0, linestyle='-', label='x1+x2')
    if xlim is None:
        plt.xlim([0, t[-1]])
    else:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


dur = 5
x1, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=1000, amp=1, freq=1.05, phase=0.0)
x2, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=1000, amp=1, freq=0.95, phase=0.8)
plot_interference(x1, x2, t, xlim=[0, dur], ylim=[-2.2, 2.2], title='Constructive Interference')

dur = 5
x1, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=1000, amp=1, freq=1.05, phase=0.0)
x2, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=1000, amp=1, freq=1.00, phase=0.4)
plot_interference(x1, x2, t, xlim=[0, dur], ylim=[-2.2, 2.2], title='Destructive Interference')

# %% [markdown]
# ## Beating
#
# Two sinusoids of similar frequency may add up or cancel out. Let f1(t) = sin(2*pi*w1*t)
# and f2(t) = sin(2*pi*w2*t) with distinct but nearby frequencies w1 ~ w2.
#
# The superposition results in a function that looks like a single sine wave with
# a slowly varying amplitude, known as **beating**.
#
# Mathematically, using a trigonometric identity:
# sin(2*pi*w1*t) + sin(2*pi*w2*t) = 2*cos(2*pi*(w1-w2)/2*t) * sin(2*pi*(w1+w2)/2*t)
#
# If (w1 - w2) is small, the cosine term has low frequency compared to the sine term.
# The result is a sine wave of frequency (w1+w2)/2 with amplitude envelope of
# frequency |w1 - w2|.

# %%
import IPython.display as ipd

Fs = 4000
dur = 5
x1, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=Fs, amp=0.5, freq=200)
x2, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=Fs, amp=0.5, freq=203)
plot_interference(x1, x2, t, ylim=[-1.1, 1.1], xlim=[0, dur],
                  title=r'Beating with beating frequency $|\omega_1-\omega_2|=3$ ($\omega_1=200, \omega_2=203$)')
plot_interference(x1, x2, t, ylim=[-1.1, 1.1], xlim=[1.115, 1.225], title=r'Zoom-in section')

ipd.display(ipd.Audio(x1 + x2, rate=Fs))

# %% [markdown]
# ## Chirp Experiment
#
# A **chirp signal** (sweep signal) has frequency that increases with time.
#
# A **linear chirp** of duration T from frequency w0 to w1 is:
# f(t) = sin(pi * (w1-w0)/T * t^2 + 2*pi*w0*t)
#
# The **instantaneous frequency** at time t is: g(t) = (w1-w0)/T * t + w0

# %%
def generate_chirp_linear(dur, freq_start, freq_end, amp=1.0, Fs=22050):
    """Generation chirp with linear frequency increase

    Args:
        dur (float): Duration (seconds)
        freq_start (float): Start frequency
        freq_end (float): End frequency
        amp (float): Amplitude
        Fs (scalar): Sampling rate

    Returns:
        x (np.ndarray): Generated chirp signal
        t (np.ndarray): Time axis (seconds)
        freq (np.ndarray): Instant frequency (Hz)
    """
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    a = (freq_end - freq_start) / dur
    freq = a * t + freq_start
    x = amp * np.sin(np.pi * a * t ** 2 + 2 * np.pi * freq_start * t)
    return x, t, freq


f_pitch = lambda p: 440 * 2 ** ((p - 69) / 12)

Fs = 4000
dur = 20
freq_start = f_pitch(57)   # A3
freq_end = f_pitch(63)     # Eb4
freq_sin = f_pitch(60)     # C4
x1, t, freq = generate_chirp_linear(dur=dur, freq_start=freq_start, freq_end=freq_end,
                                     amp=0.5, Fs=Fs)
x2, t = libfmp.c1.generate_sinusoid(dur=dur, Fs=Fs, amp=0.5, freq=freq_sin)

y = x1 + x2
ipd.display(ipd.Audio(y, rate=Fs))
plot_interference(x1, x2, t, xlim=[0, dur], ylim=[-1.1, 1.1],
                  title=r'Superposition of a linear chirp $x_1$ (A3 to E$^\flat$4) and sinusoid $x_2$ (C4)')
plot_interference(x1, x2, t, xlim=[7, 11], ylim=[-1.1, 1.1], title=r'Zoom-in section')

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Muller.

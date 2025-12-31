# %% [markdown]
# # Energy-Based Novelty
#
# Following Section 6.1.1 of [Müller, FMP, Springer 2015], we introduce in this notebook
# an energy-based approach for computing a novelty curve.

# %% [markdown]
# ## Local Energy Function
#
# Often a note onset goes along with a sudden increase of the signal's energy. Based on
# this assumption, a straightforward way to detect note onsets is to transform the signal
# into a local energy function and then to look for sudden changes in this function.
#
# The local energy of $x$ with regard to window $w$ is defined as:
#
# $$E_w^x(n) := \sum_{m=-M}^{M} |x(n+m)w(m)|^2$$

# %%
import numpy as np
import os, sys, librosa
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt
import IPython.display as ipd

sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c6

%matplotlib inline

fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_F01_Queen.csv')
ann, label_keys = libfmp.c6.read_annotation_pos(fn_ann)

fn_wav = os.path.join('..', 'data', 'C6', 'FMP_C6_F01_Queen.wav')
Fs = 22050
x, Fs = librosa.load(fn_wav, Fs)
x_duration = len(x)/Fs

N = 2048
w = signal.hann(N)

#Calculate local energy
x_square = x**2
energy_local = np.convolve(x_square, w**2, 'same')

libfmp.b.plot_signal(x, Fs, title='Waveform')
libfmp.b.plot_signal(x_square, Fs, title='Waveform (squared)')
fig, ax, line = libfmp.b.plot_signal(energy_local, Fs, color='k',
                    title='Local energy function (Hann window)')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                    nontime_axis=True, time_min=0, time_max=x_duration);
plt.tight_layout()

# %% [markdown]
# ## Discrete Derivative and Half-Wave Rectification
#
# To measure energy changes, we take a derivative of the local energy function. In the
# discrete case, we take the difference between two subsequent energy values. Furthermore,
# since we are interested in energy increases (not decreases), we apply half-wave rectification:
#
# $$|r|_{\geq 0} := \frac{r+|r|}{2}$$
#
# The energy-based novelty function is:
#
# $$\Delta_\mathrm{Energy}(n):= |E_w^x(n+1)-E_w^x(n)|_{\geq 0}$$

# %%
#Differentiation and half-wave rectification
energy_local_diff = np.diff(energy_local)
energy_local_diff = np.concatenate((energy_local_diff, np.array([0])))
novelty_energy = np.copy(energy_local_diff)
novelty_energy[energy_local_diff < 0] = 0

libfmp.b.plot_signal(energy_local_diff, Fs, color='k',
                    title='Discrete derivative (Hann window)')
fig, ax, line = libfmp.b.plot_signal(novelty_energy, Fs, color='k',
                    title='Energy-based novelty function (Hann window)')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                    nontime_axis=True, time_min=0, time_max=x_duration);
plt.tight_layout()

# %% [markdown]
# ## Effect of Window Type
#
# The smoothing effect introduced by the bell-shaped Hann window is essential. Using a
# rectangular window instead, the difference function reacts to small local fluctuations
# leading to a noisy energy function.

# %%
# Use rectangular window
w = signal.boxcar(N)
x_square = x**2
energy_local = np.convolve(x_square, w**2, 'same')
energy_local_diff = np.diff(energy_local)
energy_local_diff = np.concatenate((energy_local_diff, np.array([0])))
novelty_energy = np.copy(energy_local_diff)
novelty_energy[energy_local_diff < 0] = 0

fig, ax, line = libfmp.b.plot_signal(energy_local, Fs, color='k',
                    title='Local energy function (rectangular window)')
libfmp.b.plot_signal(energy_local_diff, Fs, color='k',
                    title='Discrete derivative (rectangular window)')
fig, ax, line = libfmp.b.plot_signal(novelty_energy, Fs, color='k', title='Energy-based novelty function (rectangular window)')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                    nontime_axis=True, time_min=0, time_max=x_duration);
plt.tight_layout()

# %% [markdown]
# ## Logarithmic Compression
#
# To account for the fact that human perception of sound intensity is logarithmic in nature,
# we apply logarithmic compression using:
#
# $$\Gamma_\gamma(v):=\log(1+ \gamma \cdot v)$$
#
# The resulting novelty function is:
#
# $$\Delta_\mathrm{Energy}^\mathrm{Log}(n):= |\Gamma_\gamma(E_w^x(n+1))-\Gamma_\gamma(E_w^x(n))|_{\geq 0}$$

# %%
def compute_novelty_energy(x, Fs=1, N=2048, H=128, gamma=10.0, norm=True):
    """Compute energy-based novelty function

    Notebook: C6/C6S1_NoveltyEnergy.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = 2048)
        H (int): Hop size (Default value = 128)
        gamma (float): Parameter for logarithmic compression (Default value = 10.0)
        norm (bool): Apply max norm (if norm==True) (Default value = True)

    Returns:
        novelty_energy (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
    """
    # x_power = x**2
    w = signal.hann(N)
    Fs_feature = Fs / H
    energy_local = np.convolve(x**2, w**2, 'same')
    energy_local = energy_local[::H]
    if gamma is not None:
        energy_local = np.log(1 + gamma * energy_local)
    energy_local_diff = np.diff(energy_local)
    energy_local_diff = np.concatenate((energy_local_diff, np.array([0])))
    novelty_energy = np.copy(energy_local_diff)
    novelty_energy[energy_local_diff < 0] = 0
    if norm:
        max_value = max(novelty_energy)
        if max_value > 0:
            novelty_energy = novelty_energy / max_value
    return novelty_energy, Fs_feature

N = 2048
H = 128
nov_1, Fs_nov = compute_novelty_energy(x, Fs=Fs, N=N, H=H, gamma=None)
nov_2, Fs_nov = compute_novelty_energy(x, Fs=Fs, N=N, H=H, gamma=1000)

fig, ax, line = libfmp.b.plot_signal(nov_1, Fs=Fs_nov, color='k',
                    title='Novelty function (original)')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                    nontime_axis=True, time_min=0, time_max=x_duration);
fig, ax, line = libfmp.b.plot_signal(nov_2, Fs=Fs_nov, color='k',
                    title='Novelty function with logarithmic compression')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                    nontime_axis=True, time_min=0, time_max=x_duration);

# %% [markdown]
# ## Example: Note Onsets for Different Instruments
#
# Energy fluctuation in nonsteady sounds as a result of vibrato or tremolo can lead to
# spurious peaks in the resulting novelty function.

# %%
fn_ann = os.path.join('..', 'data', 'C6', 'FMP_C6_F04_NoteC4_PTVF.csv')
ann, label_keys = libfmp.c6.read_annotation_pos(fn_ann, label='onset', header=0)

fn_wav = os.path.join('..', 'data', 'C6','FMP_C6_F04_NoteC4_PTVF.wav')
x, Fs = librosa.load(fn_wav)
x_duration = len(x)/Fs
N = 2048
H = 256
nov, Fs_nov = compute_novelty_energy(x, Fs=Fs, N=N, H=H, gamma=None)

plt.figure(figsize=(9,4))
ax = plt.subplot(2,1,1)
fig, ax, line = libfmp.b.plot_signal(x, Fs, ax = ax, title='Waveform')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                    nontime_axis=True, time_min=0, time_max=x_duration)

ax = plt.subplot(2,1,2)
fig, ax, line = libfmp.b.plot_signal(nov, Fs=Fs_nov, ax = ax, color='k',
                     title='Novelty function')
libfmp.b.plot_annotation_line(ann, ax=ax, label_keys=label_keys,
                nontime_axis=True, time_min=0, time_max=x_duration)
plt.ylim([0, 0.5]);
plt.tight_layout()

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Angel Villar-Corrales.

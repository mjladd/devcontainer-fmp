# %% [markdown]
# # SSM: Synthetic Generation
#
# In this notebook, we show how one may synthetically generate self-similarity matrices
# from structure annotations. In particular, based on the notions introduced in
# Section 4.2.1 of [Müller, FMP, Springer 2015], we consider SSMs with path and block
# structures.

# %% [markdown]
# ## Annotations
#
# In previous notebooks, we have looked at the Hungarian Dance No. 5 by Johannes Brahms,
# which has the musical structure A1A2B1B2CA3B3B4D.

# %%
import numpy as np
import os, sys, librosa
from matplotlib import pyplot as plt
import scipy.ndimage
import IPython.display as ipd
import pandas as pd

sys.path.append('..')
import libfmp.b
import libfmp.c4
%matplotlib inline

# Annotation
filename = 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.csv'
fn_ann = os.path.join('..', 'data', 'C4', filename)
df = libfmp.b.read_csv(fn_ann)
ipd.display(ipd.HTML(df.to_html()))

ann, color_ann = libfmp.c4.read_structure_annotation(fn_ann, fn_ann_color=filename)
fig, ax = libfmp.b.plot_segments(ann, colors=color_ann, figsize=(6, 1))
plt.xlabel('Time (seconds)');
plt.show()

print('Annotations after sampling rate conversion and removal of digits:')

ann_frames, color_ann = libfmp.c4.read_structure_annotation(fn_ann, fn_ann_color=filename,
                                                            Fs=0.5, remove_digits=True, index=True)
fig, ax = libfmp.b.plot_segments(ann_frames, colors=color_ann, figsize=(6, 1))
plt.xlabel('Time (frames)');
plt.show()

# %% [markdown]
# ## Generating SMMs from Annotations
#
# We now introduce a function that generates an SSM from a given time-discrete annotation.
# For each label one can specify if corresponding sections share a path relation, a block
# relation, or both.

# %%
def generate_ssm_from_annotation(ann, label_ann=None, score_path=1.0, score_block=0.5, main_diagonal=True,
                                 smooth_sigma=0.0, noise_power=0.0):
    """Generation of a SSM

    Notebook: C4/C4S2_SSM-Synthetic.ipynb

    Args:
        ann (list): Description of sections (see explanation above)
        label_ann (dict): Specification of property (path, block relation) (Default value = None)
        score_path (float): SSM values for occurring paths (Default value = 1.0)
        score_block (float): SSM values of blocks covering the same labels (Default value = 0.5)
        main_diagonal (bool): True if a filled main diagonal should be enforced (Default value = True)
        smooth_sigma (float): Standard deviation of a Gaussian smoothing filter.
            filter length is 4*smooth_sigma (Default value = 0.0)
        noise_power (float): Variance of additive white Gaussian noise (Default value = 0.0)

    Returns:
        S (np.ndarray): Generated SSM
    """
    N = ann[-1][1] + 1
    S = np.zeros((N, N))

    if label_ann is None:
        all_labels = [s[2] for s in ann]
        labels = list(set(all_labels))
        label_ann = {l: [True, True] for l in labels}

    for s in ann:
        for s2 in ann:
            if s[2] == s2[2]:
                if (label_ann[s[2]])[1]:
                    S[s[0]:s[1]+1, s2[0]:s2[1]+1] = score_block

                if (label_ann[s[2]])[0]:
                    length_1 = s[1] - s[0] + 1
                    length_2 = s2[1] - s2[0] + 1

                    if length_1 >= length_2:
                        scale_fac = length_2 / length_1
                        for i in range(s[1] - s[0] + 1):
                            S[s[0]+i, s2[0]+int(i*scale_fac)] = score_path
                    else:
                        scale_fac = length_1 / length_2
                        for i in range(s2[1] - s2[0] + 1):
                            S[s[0]+int(i*scale_fac), s2[0]+i] = score_path
    if main_diagonal:
        for i in range(N):
            S[i, i] = score_path
    if smooth_sigma > 0:
        S = scipy.ndimage.gaussian_filter(S, smooth_sigma)
    if noise_power > 0:
        S = S + np.sqrt(noise_power) * np.random.randn(S.shape[0], S.shape[1])
    return S

figsize = (12,3.5)
fig, ax = plt.subplots(1, 3, figsize=figsize)

S = generate_ssm_from_annotation(ann_frames, score_path=1, score_block=0.3)
libfmp.b.plot_matrix(S, ax=[ax[0]], xlabel='Time (frames)', ylabel='Time (frames)');

label_ann = {'A':[True, False], 'B':[True, False], 'C':[False, True], '' : [True, False]}
S = generate_ssm_from_annotation(ann_frames, label_ann, score_path=1, score_block=0.3, smooth_sigma=1)
libfmp.b.plot_matrix(S, ax=[ax[1]], xlabel='Time (frames)', ylabel='Time (frames)');

S = generate_ssm_from_annotation(ann_frames, score_path=1, score_block=0.2, smooth_sigma=2, noise_power=0.001)
libfmp.b.plot_matrix(S, ax=[ax[2]], xlabel='Time (frames)', ylabel='Time (frames)');
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Examples
#
# We now consider some examples that show how the function for generating SSMs can be
# used to gain a deeper understanding of the relation between musical structures and SSMs.

# %%
color_ann = {'A': [1, 0, 0, 0.2], 'B': [0, 1, 0, 0.2], 'C': [0, 0, 1, 0.2], '': [1, 1, 1, 0.2]}

figsize = (12,3.5)
fig, ax = plt.subplots(1, 3, figsize=figsize)

ann_1 = [[0, 9, 'A'], [10, 29, 'B'], [30, 49, 'B'], [50, 59, 'A'], [60, 69, 'A']]
S = generate_ssm_from_annotation(ann_1, score_path=1, score_block=0)
fig_im, ax_im, im = libfmp.b.plot_matrix(S, ax=[ax[0]], xlabel='', ylabel='', title=r'Example: $A_1B_1B_2A_2A_3$');
libfmp.b.plot_segments_overlay(ann_1, ax=ax_im[0],edgecolor='k',
                               print_labels=True, colors = color_ann, alpha=0.05)

ann_2 = [[0, 19, 'A'], [20, 34, 'A'], [35, 44, 'A'], [45, 49, 'A']]
S = generate_ssm_from_annotation(ann_2, score_path=1, score_block=0)
fig_im, ax_im, im = libfmp.b.plot_matrix(S, ax=[ax[1]], xlabel='', ylabel='', title=r'Example: $A_1A_2A_3A_4$');
libfmp.b.plot_segments_overlay(ann_2, ax=ax_im[0],edgecolor='k',
                               print_labels=True, colors = color_ann, alpha=0.05)
libfmp.b.plot_segments_overlay(ann_2, ax=ax_im[0], direction='vertical', edgecolor='k',
                               print_labels=False, colors = color_ann, alpha=0.05)


ann_3 = [[0, 9, 'A'], [10, 19, 'A'], [20, 29, 'B'], [30, 39, 'B'],
         [40, 49, 'A'], [50, 59, 'A'], [60, 69, 'B'], [70, 79, 'B'],
         [80, 99, 'C'], [100, 109, 'A'], [110, 119, 'A']]
label_ann_3 = {'A':[True, False], 'B':[True, False], 'C':[False, True]}
S = generate_ssm_from_annotation(ann_3, label_ann_3, score_path=1, score_block=0.3)
fig_im, ax_im, im = libfmp.b.plot_matrix(S, ax=[ax[2]], xlabel='', ylabel='', title=r'Example: $A_1A_2B_1B_2A_3A_4B_3B_4CA_5A_6$');
libfmp.b.plot_segments_overlay(ann_3, ax=ax_im[0],edgecolor='k',
                               print_labels=True, colors = color_ann, alpha=0.05)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Further Notes
#
# Synthetically generating and visualizing self-similarity matrices is a very instructive
# way for gaining a deeper understanding of structural properties of these matrices and
# their relation to musical annotations. Furthermore, **synthetic SSMs** are useful for
# debugging and testing automated procedures for music structure analysis. However,
# synthetic SSMs should only be used as **sanity check** and should not replace the
# evaluation based on real music examples.

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Tim Zunner.

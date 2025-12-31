# %% [markdown]
# # Experiments: Beatles Collection
#
# Following Section 5.2.4 of [Müller, FMP, Springer 2015], we conduct in this notebook some
# experiments using examples from the Beatles collection.

# %% [markdown]
# ## Beatles Collection
#
# The Beatles collection is based on twelve Beatles albums comprising 180 audio tracks in total.
# The main value of the dataset lies in the availability of high-quality **reference annotations**
# for chords, beats, key changes and music structures.

# %%
import os
import numpy as np
from matplotlib import pyplot as plt
import librosa

import sys
sys.path.append('..')
import libfmp.b
import libfmp.c3
import libfmp.c4
import libfmp.c5
%matplotlib inline

# Read annotation files
fn_ann_orig = os.path.join('..', 'data', 'C5', 'FMP_C5_Audio_Beatles_LetItBe_Beatles_1970-LetItBe-06_Chords.csv')
fn_ann = os.path.join('..', 'data', 'C5', 'FMP_C5_Audio_Beatles_LetItBe_Beatles_1970-LetItBe-06_Chords_simplified.csv')
ann_orig, _ = libfmp.c4.read_structure_annotation(fn_ann_orig)
ann, _ = libfmp.c4.read_structure_annotation(fn_ann)

# Plot chord annotations
color_ann = {'N': 'white',
             'C': 'red', 'C#': 'peru', 'D': 'orange', 'D#': 'yellow', 'Eb': 'yellow',
             'E': 'springgreen', 'F': 'cyan', 'F#': 'coral', 'G': 'blue',
             'G#': 'olive', 'A': 'teal', 'A#': 'indigo', 'Bb': 'indigo', 'B': 'pink',
             'C#:min': 'steelblue', 'C#m': 'steelblue', 'A:min': 'greenyellow', 'A:m': 'greenyellow',
             'G:min': 'olive', 'E:min': 'lightcoral', 'B:min': 'saddlebrown'}
chord_labels = libfmp.c5.get_chord_labels(ext_minor='m', nonchord=False)

libfmp.b.plot_segments(ann_orig[:7], figsize=(12, 1.2), time_label='Time (seconds)',
                        fontsize=9, colors=color_ann, alpha=0.5)
plt.title('Reference annotations with original chord labels')
plt.show()
libfmp.b.plot_segments(ann[:7], figsize=(12, 1.2), time_label='Time (seconds)',
                       fontsize=9, colors=color_ann, alpha=0.5)
plt.title('Reference annotations with reduced chord labels')
plt.show()

# %% [markdown]
# ## Song Selection
#
# We use four representative Beatles songs of comparatively short duration for our experiments.

# %%
directory =  os.path.join('..', 'data', 'C5')

song_dict = {}
song_dict[0] = ['LetItB', 'r',
                os.path.join('..', 'data', 'C5', 'FMP_C5_Audio_Beatles_LetItBe_Beatles_1970-LetItBe-06.wav'),
                os.path.join('..', 'data', 'C5', 'FMP_C5_Audio_Beatles_LetItBe_Beatles_1970-LetItBe-06_Chords_simplified.csv')]
song_dict[1] = ['HereCo', 'b',
                os.path.join('..', 'data', 'C5', 'FMP_C5_Audio_Beatles_HereComesTheSun_Beatles_1969-AbbeyRoad-07.wav'),
                os.path.join('..', 'data', 'C5', 'FMP_C5_Audio_Beatles_HereComesTheSun_Beatles_1969-AbbeyRoad-07_Chords_simplified.csv')]
song_dict[2] = ['ObLaDi', 'c',
                os.path.join('..', 'data', 'C5', 'FMP_C5_Audio_Beatles_ObLaDiObLaDa_Beatles_1968-TheBeatlesTheWhiteAlbumDisc1-04.wav'),
                os.path.join('..', 'data', 'C5', 'FMP_C5_Audio_Beatles_ObLaDiObLaDa_Beatles_1968-TheBeatlesTheWhiteAlbumDisc1-04_Chords_simplified.csv')]
song_dict[3] = ['PennyL', 'g',
                os.path.join('..', 'data', 'C5', 'FMP_C5_Audio_Beatles_PennyLane_Beatles_1967-MagicalMysteryTour-09.wav'),
                os.path.join('..', 'data', 'C5', 'FMP_C5_Audio_Beatles_PennyLane_Beatles_1967-MagicalMysteryTour-09_Chords_simplified.csv')]

chord_labels = libfmp.c5.get_chord_labels(ext_minor='m', nonchord=False)
song_selected = [0, 1, 2, 3]

for s in song_selected:
    song_id = song_dict[s][0]
    fn_ann = song_dict[s][3]
    fn_ann_orig = fn_ann.replace('_simplified','')

    ann_orig, _ = libfmp.c4.read_structure_annotation(fn_ann_orig)
    ann, _ = libfmp.c4.read_structure_annotation(fn_ann)

    libfmp.b.plot_segments(ann_orig[:30], figsize=(12, 1),
                           fontsize=9, colors=color_ann, alpha=0.5)
    plt.title('Song: %s' % song_id)
    plt.show()
    libfmp.b.plot_segments(ann[:30], figsize=(12, 1.2), time_label='Time (seconds)',
                           fontsize=9, colors=color_ann, alpha=0.5)
    plt.show()

# %% [markdown]
# ## Feature Extraction
#
# We compute three different chroma types: STFT-based, CQT-based, and IIR-based chroma features.

# %%
def compute_X_dict(song_selected, version='STFT', details=True):
    X_dict = {}
    Fs_X_dict = {}
    ann_dict = {}
    x_dur_dict = {}
    chord_labels = libfmp.c5.get_chord_labels(ext_minor='m', nonchord=False)
    for s in song_selected:
        if details is True:
            print('Processing: ', song_dict[s][0])
        fn_wav = song_dict[s][2]
        fn_ann = song_dict[s][3]
        N = 2048
        H = 1024
        if version == 'STFT':
            X, Fs_X, x, Fs, x_dur = \
                libfmp.c5.compute_chromagram_from_filename(fn_wav, N=N, H=H, gamma=0.1, version='STFT')
        if version == 'CQT':
            X, Fs_X, x, Fs, x_dur = \
                libfmp.c5.compute_chromagram_from_filename(fn_wav, H=H, version='CQT')
        if version == 'IIR':
            X, Fs_X, x, Fs, x_dur = \
                libfmp.c5.compute_chromagram_from_filename(fn_wav, N=N, H=H, gamma=10, version='IIR')
        X_dict[s] = X
        Fs_X_dict[s] = Fs_X
        x_dur_dict[s] = x_dur
        N_X = X.shape[1]
        ann_dict[s] = libfmp.c5.convert_chord_ann_matrix(fn_ann, chord_labels, Fs=Fs_X, N=N_X, last=False)
    return X_dict, Fs_X_dict, ann_dict, x_dur_dict, chord_labels

song_selected = [0, 1, 2, 3]
print('===== Computation of STFT-based chromagrams =====')
X_dict_STFT, Fs_X_dict_STFT, ann_dict_STFT, x_dur_dict, chord_labels = compute_X_dict(song_selected, version='STFT')
print('===== Computation of CQT-based chromagrams =====')
X_dict_CQT, Fs_X_dict_CQT, ann_dict_CQT, x_dur_dict, chord_labels = compute_X_dict(song_selected, version='CQT')
print('===== Computation of IIR-based chromagrams =====')
X_dict_IIR, Fs_X_dict_IIR, ann_dict_IIR, x_dur_dict, chord_labels = compute_X_dict(song_selected, version='IIR')

# %%
cmap = 'gray_r'
for s in song_selected:
    fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1],
                                              'height_ratios': [2]}, figsize=(10, 2.5))
    title = '%s, STFT (%0.1f Hz)' % (song_dict[s][0], Fs_X_dict_STFT[s])
    libfmp.b.plot_chromagram(X_dict_STFT[s], Fs=Fs_X_dict_CQT[s], ax=[ax[0]],
                             chroma_yticks=[0, 4, 7, 11], clim=[0, 1], cmap=cmap,
                             title=title, ylabel='Chroma', colorbar=True, xlim=[0, 15])

    title = '%s, CQT (%0.1f Hz)' % (song_dict[s][0], Fs_X_dict_CQT[s])
    libfmp.b.plot_chromagram(X_dict_CQT[s], Fs=Fs_X_dict_CQT[s], ax=[ax[1]],
                             chroma_yticks=[0, 4, 7, 11], clim=[0, 1], cmap=cmap,
                             title=title, ylabel='Chroma', colorbar=True, xlim=[0, 15])

    title = '%s, IIR (%0.1f Hz)' % (song_dict[s][0], Fs_X_dict_IIR[s])
    libfmp.b.plot_chromagram(X_dict_IIR[s], Fs=Fs_X_dict_IIR[s], ax=[ax[2]],
                             chroma_yticks=[0, 4, 7, 11], clim=[0, 1], cmap=cmap,
                             title=title, ylabel='Chroma', colorbar=True, xlim=[0, 15])
    plt.tight_layout()

# %% [markdown]
# ## Chord Recognition Procedures
#
# We apply template-based and HMM-based chord recognition to the chromagrams.

# %%
def chord_recognition_all(X, ann_matrix, p=0.15, filt_len=None, filt_type='mean'):
    """Conduct template- and HMM-based chord recognition and evaluates the approaches

    Notebook: C5/C5S3_ChordRec_Beatles.ipynb

    Args:
        X (np.ndarray): Chromagram
        ann_matrix (np.ndarray): Reference annotation as given as time-chord binary matrix
        p (float): Self-transition probability used for HMM (Default value = 0.15)
        filt_len (int): Filter length used for prefilitering (Default value = None)
        filt_type (str): Filter type used for prefilitering (Default value = 'mean')

    Returns:
        result_Tem (tuple): Chord recogntion evaluation results ([P, R, F, TP, FP, FN]) for template-based approach
        result_HMM (tuple): Chord recogntion evaluation results ([P, R, F, TP, FP, FN]) for HMM-based approach
        chord_Tem (np.ndarray): Template-based chord recogntion result given as binary matrix
        chord_HMM (np.ndarray): HMM-based chord recogntion result given as binary matrix
        chord_sim (np.ndarray): Chord similarity matrix
    """
    if filt_len is not None:
        if filt_type == 'mean':
            X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(X, Fs=1, filt_len=filt_len, down_sampling=1)
        if filt_type == 'median':
            X, Fs_X = libfmp.c3.median_downsample_feature_sequence(X, Fs=1, filt_len=filt_len, down_sampling=1)
    # Template-based chord recogntion
    chord_sim, chord_Tem = libfmp.c5.chord_recognition_template(X, norm_sim='1')
    result_Tem = libfmp.c5.compute_eval_measures(ann_matrix, chord_Tem)
    # HMM-based chord recogntion
    A = libfmp.c5.uniform_transition_matrix(p=p)
    C = 1 / 24 * np.ones((1, 24))
    B_O = chord_sim
    chord_HMM, _, _, _ = libfmp.c5.viterbi_log_likelihood(A, C, B_O)
    result_HMM = libfmp.c5.compute_eval_measures(ann_matrix, chord_HMM)
    return result_Tem, result_HMM, chord_Tem, chord_HMM, chord_sim

def plot_chord_recognition_result(ann_matrix, result, chord_matrix, chord_labels,
                                  xlim=None, Fs_X=1, title='', figsize=(12, 4)):
    P, R, F, TP, FP, FN = result
    method='HMM'
    title = title + ' (TP=%d, FP=%d, FN=%d, P=%.3f, R=%.3f, F=%.3f)' % (TP, FP, FN, P, R, F)
    fig, ax, im = libfmp.c5.plot_matrix_chord_eval(ann_matrix, chord_matrix, Fs=Fs_X, figsize=figsize,
                         title=title, ylabel='Chord', xlabel='Time (frames)', chord_labels=chord_labels)
    if xlim is not None:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.show()

song_selected = [0]
for s in song_selected:
    output = chord_recognition_all(X_dict_STFT[s], ann_dict_STFT[s][0], p=0.15)
    result_Tem, result_HMM, chord_Tem, chord_HMM, chord_sim = output
    title = 'Song: %s [STFT; Template]' % song_dict[s][0]
    plot_chord_recognition_result(ann_dict_STFT[s][0], result_Tem, chord_Tem, chord_labels, title=title)
    title = 'Song: %s [STFT; HMM]' % song_dict[s][0]
    plot_chord_recognition_result(ann_dict_STFT[s][0], result_HMM, chord_HMM, chord_labels, title=title)

    output = chord_recognition_all(X_dict_CQT[s], ann_dict_CQT[s][0], p=0.15)
    result_Tem, result_HMM, chord_Tem, chord_HMM, chord_sim = output
    title = 'Song: %s [CQT; Template]' % song_dict[s][0]
    plot_chord_recognition_result(ann_dict_CQT[s][0], result_Tem, chord_Tem, chord_labels, title=title)
    title = 'Song: %s [CQT; HMM]' % song_dict[s][0]
    plot_chord_recognition_result(ann_dict_STFT[s][0], result_HMM, chord_HMM, chord_labels, title=title)

    output = chord_recognition_all(X_dict_IIR[s], ann_dict_IIR[s][0], p=0.15)
    result_Tem, result_HMM, chord_Tem, chord_HMM, chord_sim = output
    title = 'Song: %s [IIR; Template]' % song_dict[s][0]
    plot_chord_recognition_result(ann_dict_STFT[s][0], result_Tem, chord_Tem, chord_labels, title=title)
    title = 'Song: %s [IIR; HMM]' % song_dict[s][0]
    plot_chord_recognition_result(ann_dict_IIR[s][0], result_HMM, chord_HMM, chord_labels, title=title)

# %%
song_selected = [0, 1, 2, 3]
for s in song_selected:
    output = chord_recognition_all(X_dict_CQT[s], ann_dict_CQT[s][0], p=0.15)
    result_Tem, result_HMM, chord_Tem, chord_HMM, chord_sim = output
    title='Song: %s [CQT; HMM]' % song_dict[s][0]
    plot_chord_recognition_result(ann_dict_STFT[s][0], result_HMM, chord_HMM, chord_labels, title=title)

# %% [markdown]
# ## Prefiltering Experiment
#
# We investigate the effect of temporal smoothing on the chord recognition accuracy.

# %%
def compute_mean_result(result_dict, song_selected):
    S = len(song_selected)
    result_mean =  np.copy(result_dict[song_selected[0]])
    for s in range(1, S):
        result_mean = result_mean + result_dict[song_selected[s]]
    result_mean = result_mean / S
    return result_mean

def plot_statistics(para_list, song_dict, song_selected, result_dict, ax,
                    ylim=None, title='', xlabel='', ylabel='F-measure', legend=True):
    for s in song_selected:
        color = song_dict[s][1]
        song_id = song_dict[s][0]
        ax.plot(para_list, result_dict[s], color=color,
                linestyle=':', linewidth='2', label=song_id)
    ax.plot(para_list, compute_mean_result(result_dict, song_selected), color='k',
            linestyle='-',linewidth='2', label='Mean')
    if legend==True:
        ax.legend(loc='upper right')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlim([para_list[0], para_list[-1]])

def experiment_chord_recognition(song_selected, song_dict, X_dict, ann_dict,
                                 para_list, para_type=None, p=0.15,
                                 filt_len=None, filt_type='mean', detail=True):
    M = len(para_list)
    result_F_Tem = np.zeros(M)
    result_F_HMM = np.zeros(M)
    result_F_Tem_dict = {}
    result_F_HMM_dict = {}
    for s in song_selected:
        if detail is True:
            print('Processing:', song_dict[s][0])
        for m in range(M):
            if para_type == 'smooth':
                filt_len = para_list[m]
            if para_type == 'p':
                p = para_list[m]
            output = chord_recognition_all(X_dict[s], ann_dict[s][0],
                                           filt_len=filt_len, filt_type=filt_type, p=p)
            result_Tem, result_HMM, chord_Tem, chord_HMM, chord_sim = output
            result_F_Tem[m] = result_Tem[2]
            result_F_HMM[m] = result_HMM[2]
        result_F_Tem_dict[s] = np.copy(result_F_Tem)
        result_F_HMM_dict[s] = np.copy(result_F_HMM)
    return result_F_Tem_dict, result_F_HMM_dict

song_selected = [0, 1, 2, 3]
para_list = np.arange(32) * 2 + 1
print('===== Prefiltering experiment using STFT-based chromagrams =====')
result_STFT = experiment_chord_recognition(song_selected, song_dict, X_dict_STFT,
                                           ann_dict_STFT, para_list, para_type='smooth',
                                           p=0.15, filt_len=None, filt_type='mean')
print('===== Prefiltering experiment using CQT-based chromagrams =====')
result_CQT  = experiment_chord_recognition(song_selected, song_dict, X_dict_CQT,
                                           ann_dict_CQT, para_list, para_type='smooth',
                                           p=0.15, filt_len=None, filt_type='mean')
print('===== Prefiltering experiment using IIR-based chromagrams =====')
result_IIR  = experiment_chord_recognition(song_selected, song_dict, X_dict_IIR,
                                           ann_dict_IIR, para_list, para_type='smooth',
                                           p=0.15, filt_len=None, filt_type='mean')

# Plot result
fig, ax = plt.subplots(2, 3, figsize=(11, 7))
xlabel='Smoothing length (frames)'
ylim = [0.3, 0.95]
title='STFT, Template'
plot_statistics(para_list, song_dict, song_selected, result_STFT[0], ax[0, 0],
                ylim=ylim, title=title, xlabel=xlabel)
title='STFT, HMM'
plot_statistics(para_list, song_dict, song_selected, result_STFT[1], ax[1, 0],
                ylim=ylim, title=title, xlabel=xlabel)
title='CQT, Template'
plot_statistics(para_list, song_dict, song_selected, result_CQT[0], ax[0, 1],
                ylim=ylim, title=title, xlabel=xlabel)
title='CQT, HMM'
plot_statistics(para_list, song_dict, song_selected, result_CQT[1], ax[1, 1],
                ylim=ylim, title=title, xlabel=xlabel)
title='IIR, Template'
plot_statistics(para_list, song_dict, song_selected, result_IIR[0], ax[0, 2],
                ylim=ylim, title=title, xlabel=xlabel)
title='IIR, HMM'
plot_statistics(para_list, song_dict, song_selected, result_IIR[1], ax[1, 2],
                ylim=ylim, title=title, xlabel=xlabel)
plt.tight_layout()

# %% [markdown]
# ## Self-Transition Probability Experiment
#
# We investigate the effect of self-transition probabilities on the chord recognition accuracy.

# %%
song_selected = [0, 1, 2, 3]
para_list = (np.arange(51)) * 0.02
print('===== Self-transition probability experiment using STFT-based chromagrams =====')
result_STFT = experiment_chord_recognition(song_selected, song_dict, X_dict_STFT, ann_dict_STFT,
                                           para_list, para_type='p', p=0.15, filt_len=None)
print('===== Self-transition probability experiment using CQT-based chromagrams =====')
result_CQT  = experiment_chord_recognition(song_selected, song_dict, X_dict_CQT, ann_dict_CQT,
                                           para_list, para_type='p', p=0.15, filt_len=None)
print('===== Self-transition probability experiment using IIR-based chromagrams =====')
result_IIR  = experiment_chord_recognition(song_selected, song_dict, X_dict_IIR, ann_dict_IIR,
                                           para_list, para_type='p', p=0.15, filt_len=None)

# Plot result
fig, ax = plt.subplots(1,3, figsize=(11, 3.5))
xlabel='Transition probability'
ylim = [0.3, 0.95]
title='STFT, HMM'
plot_statistics(para_list, song_dict, song_selected, result_STFT[1], ax[0], ylim=ylim, title=title, xlabel=xlabel)
title='CQT, HMM'
plot_statistics(para_list, song_dict, song_selected, result_CQT[1], ax[1], ylim=ylim, title=title, xlabel=xlabel)
title='IIR, HMM'
plot_statistics(para_list, song_dict, song_selected, result_IIR[1], ax[2], ylim=ylim, title=title, xlabel=xlabel)
plt.tight_layout()

# %% [markdown]
# ## Musical Relevance: Class Imbalance
#
# An **imbalance in the class distribution** may lead to surprisingly high evaluation measures,
# which does not say anything about the actual performance of the underlying classification approach.

# %%
song_selected = [2]
for s in song_selected:
    output = chord_recognition_all(X_dict_CQT[s], ann_dict_CQT[s][0], p=0.15)
    result_Tem, result_HMM, chord_Tem, chord_HMM, chord_sim = output
    chord_dull = np.zeros(chord_Tem.shape)
    chord_dull[10, :] = 1
    result_dull = libfmp.c5.compute_eval_measures(ann_dict_CQT[s][0], chord_dull)

    title='Song: %s [CQT; Dull]' % song_dict[s][0]
    plot_chord_recognition_result(ann_dict_CQT[s][0], result_dull,
                                  chord_dull, chord_labels, title=title)
    title='Song: %s [CQT; Template]' % song_dict[s][0]
    plot_chord_recognition_result(ann_dict_CQT[s][0], result_Tem,
                                  chord_Tem, chord_labels, title=title)
    title='Song: %s [CQT; HMM]' % song_dict[s][0]
    plot_chord_recognition_result(ann_dict_CQT[s][0], result_HMM,
                                  chord_HMM, chord_labels, title=title)

# %% [markdown]
# ## Musical Relevance: Chord Label Reduction
#
# Another major issue in our chord recognition approach is the limitation to only 24 chord
# classes corresponding to the 12 major and 12 minor triads.

# %%
for s in [3]:
    song_id = song_dict[s][0]
    fn_ann = song_dict[s][3]
    fn_ann_orig = fn_ann.replace('_simplified', '')

    # Read original annotation (Harte) and triad-reduced annotation
    ann_orig, _ = libfmp.c4.read_structure_annotation(fn_ann_orig)
    ann, _ = libfmp.c4.read_structure_annotation(fn_ann)
    # Replace in original annotation all labels that were reducued by non-chord label
    for k in range(len(ann)):
        if ann[k][2] != ann_orig[k][2]:
            ann_orig[k][2] = 'N'

    libfmp.b.plot_segments(ann, figsize=(12, 1.2), time_label='Time (seconds)',
                           print_labels=False, colors=color_ann, alpha=1)
    plt.title('Song: %s: Triad reduction' % song_id)
    plt.show()
    libfmp.b.plot_segments(ann_orig, figsize=(12, 1),
                           print_labels=False, colors=color_ann, alpha=1)
    plt.title('Song: %s: Non-chord reduction' % song_id)
    plt.show()

# %%
song_selected = [3]
for s in song_selected:
    output = chord_recognition_all(X_dict_CQT[s], ann_dict_CQT[s][0], p=0.15)
    result_Tem, result_HMM, chord_Tem, chord_HMM, chord_sim = output
    song_id = song_dict[s][0]
    fn_ann = song_dict[s][3]
    fn_ann_orig = fn_ann.replace('_simplified', '')
    ann_orig, _ = libfmp.c4.read_structure_annotation(fn_ann_orig)
    output =  libfmp.c5.convert_chord_ann_matrix(fn_ann_orig, chord_labels, Fs=Fs_X_dict_CQT[s],
                                                 N=X_dict_CQT[s].shape[1], last=False)
    ann_matrix_N = output[0]
    output = chord_recognition_all(X_dict_CQT[s], ann_matrix_N, p=0.15)
    result_Tem_N, result_HMM_N, chord_Tem_N, chord_HMM_N, chord_sim_N = output
    title = 'Triad reduction: %s [CQT; HMM] ' % song_dict[s][0]
    plot_chord_recognition_result(ann_dict_CQT[s][0], result_HMM, chord_HMM, chord_labels, title=title)
    title = 'Non-chord label: %s [CQT; HMM] ' % song_dict[s][0]
    plot_chord_recognition_result(ann_matrix_N, result_HMM_N, chord_HMM_N, chord_labels, title=title)

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller and Christof Weiß.

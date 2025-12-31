# %% [markdown]
# # Chapter 5: Chord Recognition
#
# In Chapter 5 of [Müller, FMP, Springer 2015], we consider the problem of analyzing
# harmonic properties of a piece of music by determining a descriptive progression of
# chords from a given audio recording. We take this opportunity to first discuss some
# basic theory of harmony including concepts such as intervals, chords, and scales.
# Then, motivated by the automated chord recognition scenario, we introduce template-based
# matching procedures and hidden Markov models—a concept of central importance for the
# analysis of temporal patterns in time-dependent data streams including speech, gestures,
# and music.
#
# 5.1 Basic Theory of Harmony
# 5.2 Template-Based Chord Recognition
# 5.3 HMM-Based Chord Recognition
# 5.4 Further Notes

# %% [markdown]
# ## Notebooks
#
# - Intervals: Semitone; cent; equal-tempered scale; enharmonic equivalence; harmonic series;
#   interval; unison; octave; fifth; pure interval; just interval; consonance; dissonance
# - Chords: Chord; dyad; triad; tetrad; major; minor; diminished; augmented triad; root;
#   inversion; seventh chord
# - Musical Scales and Circle of Fifths: Musical scale; scale step; chromatic scale; half step;
#   whole step; major scale; minor scale; scale degree; tonic; dominant; subdominant;
#   diatonic scale; circle of fifths; musical key
# - Template-Based Chord Recognition: Prefiltering; postfiltering; major triad; minor triad;
#   chroma feature; template; chord label; time-chord representation
# - Chord Recognition Evaluation: Ground truth; label; correct; incorrect; accuracy;
#   true positive; false positive; false negative; chord ambiguity; major-minor confusion
# - Hidden Markov Model (HMM): Markov chain; state; Markov property; state transition
#   probability; discrete HMM; observation symbol; emission probability
# - Viterbi Algorithm: Uncovering problem; dynamic programming; Viterbi algorithm;
#   implementation; logarithmic domain
# - HMM-Based Chord Recognition: Discrete HMM; codebook; self-transition;
#   transposition-invariant transition matrix; uniform transition matrix
# - Experiments: Beatles Collection: Beatles collection; annotation; reference;
#   chord label reduction; chroma feature; template-based approach; HMM-based approach

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

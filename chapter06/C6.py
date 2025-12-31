# %% [markdown]
# # Chapter 6: Tempo and Beat Tracking
#
# Tempo and beat are further fundamental properties of music. In Chapter 6 of
# [Müller, FMP, Springer 2015], we introduce the basic ideas on how to extract
# tempo-related information from audio recordings. In this scenario, a first
# challenge is to locate note onset information—a task that requires methods
# for detecting changes in energy and spectral content. To derive tempo and
# beat information, note onset candidates are then analyzed with regard to
# quasiperiodic patterns. This leads us to the study of general methods for
# local periodicity analysis of time series.
#
# 6.1 Onset Detection
# 6.2 Tempo Analysis
# 6.3 Beat and Pulse Tracking
# 6.4 Further Notes

# %% [markdown]
# ## Notebooks
#
# - **Onset Detection**: Transient; attack; onset; novelty function; peak picking
# - **Energy-Based Novelty**: Local energy; half-wave rectification; novelty function
# - **Spectral-Based Novelty**: Spectral flux; logarithmic compression
# - **Phase-Based Novelty**: Phase wrapping; phase unwrapping; principal argument function
# - **Complex-Domain Novelty**: Phase; magnitude
# - **Novelty Comparison**: Comparison of approaches
# - **Peak Picking**: Local maximum; smoothing; adaptive thresholding
# - **Tempo and Beat**: Tempogram; pulse level; tempo octave
# - **Fourier Tempogram**: Fourier analysis; windowed sinusoid; tempo resolution
# - **Autocorrelation Tempogram**: Short-time autocorrelation; time-lag representation
# - **Cyclic Tempogram**: Scaling parameter; tempo octave; tempo harmonic
# - **Predominant Local Pulse**: Local pulse; tempo range; pulse levels
# - **Beat Tracking by Dynamic Programming**: Penalty function; beat sequence
# - **Adaptive Windowing**: Beat-synchronous feature; transient removal

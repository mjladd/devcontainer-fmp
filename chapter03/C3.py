# %% [markdown]
# # Chapter 3: Music Synchronization
#
# In Chapter 3 of [Müller, FMP, Springer 2015], we study the problem of music
# synchronization. The objective is to temporally align compatible representations
# of the same piece of music. Considering this scenario, we explain the need for
# musically informed audio features. In particular, we introduce the concept of
# chroma-based music features, which capture properties that are related to harmony
# and melody. Furthermore, we study an alignment technique known as dynamic time
# warping (DTW), a concept that is applicable for the analysis of general time series.
# For its efficient computation, we discuss an algorithm based on dynamic programming—
# a widely used method for solving a complex problem by breaking it down into a
# collection of simpler subproblems.
#
# 3.1 Audio Features
# 3.2 Dynamic Time Warping
# 3.3 Applications
# 3.4 Further Notes

# %% [markdown]
# ## Notebooks
#
# - Log-Frequency Spectrogram and Chromagram
# - Logarithmic Compression
# - Feature Normalization
# - Temporal Smoothing and Downsampling
# - Tuning and Transposition
# - Dynamic Time Warping (DTW)
# - DTW Variants
# - Music Synchronization
# - Application: Music Navigation
# - Application: Tempo Curves

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

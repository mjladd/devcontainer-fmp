# %% [markdown]
# # Chapter 4: Music Structure Analysis
#
# In Chapter 4 of [Müller, FMP, Springer 2015], we address a central and well-researched
# area within MIR known as music structure analysis. Given a music recording, the objective
# is to identify important structural elements and to temporally segment the recording
# according to these elements. Within this scenario, we discuss fundamental segmentation
# principles based on repetitions, homogeneity, and novelty—principles that also apply to
# other types of multimedia beyond music. As an important technical tool, we study in
# detail the concept of self-similarity matrices and discuss their structural properties.
# Finally, we briefly touch the topic of evaluation, introducing the notions of precision,
# recall, and F-measure. These measures are used to compare the computed results that are
# obtained by an automated procedure with so-called ground truth annotations that are
# typically generated manually by some domain expert.
#
# 4.1 General Principles
# 4.2 Self-Similarity Matrices
# 4.3 Audio Thumbnailing
# 4.4 Novelty-Based Segmentation
# 4.5 Evaluation
# 4.6 Further Notes

# %% [markdown]
# ## Notebooks
#
# * Music Structure Analysis: General Principles [Section 4.1]
# * Self-Similarity Matrix (SSM) [Section 4.2.1]
# * SSM: Synthetic Generation [Section 4.2.1, Exercise 4.9]
# * SSM: Feature Smoothing [Section 4.2.2.1]
# * SSM: Path Enhancement [Section 4.2.2.2]
# * SSM: Transposition Invariance [Section 4.2.2.3]
# * SSM: Thresholding [Section 4.2.2.4, Exercise 4.5]
# * Audio Thumbnailing [Section 4.3]
# * Scape Plot Representation [Section 4.3.2, Exercise 4.12]
# * Novelty-Based Segmentation [Section 4.4.1]
# * Structure Feature [Section 4.4.2, Exercise 4.13]
# * Evaluation [Section 4.5]

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

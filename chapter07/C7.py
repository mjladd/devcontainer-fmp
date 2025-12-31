# %% [markdown]
# # Chapter 7: Content-Based Audio Retrieval
#
# One important topic in music information retrieval is concerned with the development of search
# engines that enable users to explore music collections in a flexible and intuitive way. In
# Chapter 7 of [Müller, FMP, Springer 2015], we discuss audio retrieval strategies that follow
# the query-by-example paradigm: given an audio query, the task is to retrieve all documents
# that are somehow similar or related to the query. Starting with audio identification, a
# technique used in many commercial applications such as Shazam, we study various retrieval
# strategies to handle different degrees of similarity. Furthermore, considering efficiency
# issues, we discuss fundamental indexing techniques based on inverted lists—a concept
# originally used in text retrieval.
#
# 7.1 Audio Identification
# 7.2 Audio Matching
# 7.3 Version Identification
# 7.4 Further Notes

# %% [markdown]
# ## Notebooks
#
# - Content-Based Audio Retrieval: Information retrieval; query; document; item; content;
#   query-by-example; specificity; granularity; audio identification; fingerprinting;
#   audio matching; version identification
#
# - Audio Identification: Audio fingerprint; client-server model; specificity; robustness;
#   compactness; scalability; spectral peaks; constellation map; indexing; hash; peak pairs
#
# - Feature Design (Chroma, CENS): Chromagram; normalization; smoothing; downsampling;
#   quantization
#
# - Diagonal Matching: Matching function; cost matrix; dot product; local minimum; retrieval;
#   match; multiple-query strategy; scaled version; tempo variation
#
# - Subsequence DTW: Local alignment; dynamic time warping; cost matrix; accumulated cost
#   matrix; matching function
#
# - Audio Matching: Chroma features; CENS; matching function; match; cyclic shift;
#   transposition-invariant matching function
#
# - Common Subsequence Matching: Sequence alignment; global alignment; local alignment;
#   score matrix; path; step size; induced segment; accumulated score matrix; dynamic
#   programming; backtracking
#
# - Version Identification: Cover song; document-level retrieval; tonal properties; chroma
#   features; local alignment; common longest subsequence; similarity score
#
# - Evaluation Measures: Document-level retrieval; item; similarity score; rank; top rank;
#   relevance function; precision; recall; PR curve; break-even point; maximal F-measure;
#   average precision; mean average precision (MAP)

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Meinard Müller.

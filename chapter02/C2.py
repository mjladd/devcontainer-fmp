# %% [markdown]
# # Chapter 2: Fourier Analysis of Signals
#
# The Fourier transform is perhaps the most fundamental tool in signal processing.
# Chapter 2 of [Muller, FMP, Springer 2015] approaches Fourier analysis from various
# perspectives and discusses their most important properties in the context of audio
# signal processing. In particular, the notion of a spectrogram, which yields a
# time-frequency representation of an audio signal, is introduced.
#
# 2.1 The Fourier Transform in a Nutshell
# 2.2 Signals and Signal Spaces
# 2.3 Fourier Transform
# 2.4 Discrete Fourier Transform (DFT)
# 2.5 Short-Time Fourier Transform (STFT)
# 2.6 Further Notes

# %% [markdown]
# ## Notebooks
#
# - Complex Numbers: Absolute value; angle; polar representation; conjugation; inverse
# - Exponential Function: Power series; Euler's formula; root of unity
# - Discrete Fourier Transform (DFT): Inner product; DFT matrix; FFT; runtime experiments
# - DFT: Phase: Exponential function; polar coordinates; complex Fourier coefficient
# - STFT: Time localization; spectrogram; physical interpretation
# - STFT: Window Function: Window type; window size
# - STFT: Padding: Padding variants; edge phenomena
# - STFT: Frequency Grid Density: DFT frequency grid; zero padding
# - STFT: Frequency Interpolation: Linear/cubic interpolation; log-frequency STFT
# - STFT: Inverse: DFT; inverse DFT; window function; overlap-add technique
# - STFT: Conventions and Implementations: Time axis convention; centered windowing
# - Digital Signals: Sampling: Equidistant sampling; aliasing; signal reconstruction
# - Digital Signals: Quantization: Uniform quantization; quantization error
# - Interference and Beating: Constructive/destructive interference; chirp; sweep

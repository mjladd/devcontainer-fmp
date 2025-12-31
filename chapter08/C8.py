# %% [markdown]
# <div>
# <a href="http://www.music-processing.de/"><img style="float:left;" src="../data/FMP_Teaser_Cover.png" width=40% alt="FMP"></a>
# <a href="https://www.audiolabs-erlangen.de"><img src="../data/Logo_AudioLabs_Long.png" width=59% style="float: right;" alt="AudioLabs"></a>
# </div>

# %% [markdown]
# <div>
# <a href="../C8/C8.html"><img src="../data/C8_nav.png" width="100"  style="float:right;" alt="C8"></a>
# <h1> Chapter 8: Musically Informed Audio Decomposition</h1> 
# </div>
# 
# <br/>
# 
# <p>
# In Chapter 8 of <a href="http://www.music-processing.de">[Müller, FMP, Springer 2015]</a> on audio decomposition, we present a challenging research direction that is closely related to source separation. Within this wide research area, we consider three subproblems: harmonic&ndash;percussive separation, main melody extraction, and score-informed audio decomposition. Within these scenarios, we discuss a number of key techniques including instantaneous frequency estimation, fundamental frequency (F0) estimation, spectrogram inversion, and nonnegative matrix factorization (NMF). Furthermore, we encounter a number of acoustic and musical properties of audio recordings that have been introduced and discussed in previous chapters.
# </p>
# 
# <p>
# 8.1 Harmonic&ndash;Percussive Separation <br />
# 8.2 Melody Extraction <br />
# 8.3 NMF-Based Audio Decomposition <br />
# 8.4 Further Notes
# </p>
# 

# %% [markdown]
# ## Notebooks
# 
# <table class="table table-hover" style="border:none; font-size: 90%; width:100%; text-align:left">
# <colgroup>
#     <col span="1" style="width:30%; text-align:left">
#     <col span="1" style="width:50%; text-align:left">
#     <col span="1" style="width:10%; text-align:left">
#     <col span="1" style="width:10%; text-align:left">   
# </colgroup>
# <tr text-align="left" style="border:1px solid #C8C8C8; background-color:#F0F0F0" >
#     <td style="border:none; text-align:left"><b>Topic</b></td>
#     <td style="border:none; text-align:left"><b>Relation to <a href="http://www.music-processing.de">[Müller, FMP, Springer 2015]</a> & Description</a></b></td> 
#     <td style="border:none; text-align:left"><b>HTML</b></td>
#     <td style="border:none; text-align:left"><b>IPYNB</b></td>
# </tr>
# 
# <tr text-align="left" style="border:1px solid #C8C8C8">
#     <td style="border:none; text-align:left"><a href="../C8/C8S1_HPS.html">Harmonic&ndash;Percussive Separation (HPS)</a></td>
#     <td style="border:none; text-align:left">[Section 8.1.1]<br>Harmonic sound; percussive sound; median filter; binary mask; soft mask; signal reconstruction; HPR experiments; Violin&ndash;Castanets example; audio examples (diverse)</td> 
#     <td style="border:none; text-align:left"><a href="../C8/C8S1_HPS.html">[html]</a></td>      
#     <td style="border:none; text-align:left"><a href="../C8/C8S1_HPS.ipynb">[ipynb]</a></td>    
# </tr>
# 
# <tr text-align="left" style="border:1px solid #C8C8C8">
#     <td style="border:none; text-align:left"><a href="../C8/C8S1_HRPS.html">Harmonic&ndash;Residual&ndash;Percussive Separation (HRPS)</a></td>
#     <td style="border:none; text-align:left">[Section 8.1.1, Exercise 8.5]<br>Separation factor; residual component; binary mask; cascaded HRPS; Violin&ndash;Applause&ndash;Castanets example; Bornemark example (Stop Messing With Me)</td> 
#     <td style="border:none; text-align:left"><a href="../C8/C8S1_HRPS.html">[html]</a></td>      
#     <td style="border:none; text-align:left"><a href="../C8/C8S1_HRPS.ipynb">[ipynb]</a></td>    
# </tr>
# 
# <tr text-align="left" style="border:1px solid #C8C8C8">
#     <td style="border:none; text-align:left"><a href="../C8/C8S1_SignalReconstruction.html">Signal Reconstruction</a></td>
#     <td style="border:none; text-align:left">[Section 8.1.2]<br>Inverse DFT; inverse STFT; modified STFT; overlap&ndash;add procedure; Griffin&ndash;Lim optimization problem</td> 
#     <td style="border:none; text-align:left"><a href="../C8/C8S1_SignalReconstruction.html">[html]</a></td>      
#     <td style="border:none; text-align:left"><a href="../C8/C8S1_SignalReconstruction.ipynb">[ipynb]</a></td>    
# </tr>
# 
# 
# <tr text-align="left" style="border:1px solid #C8C8C8">
#     <td style="border:none; text-align:left"><a href="../C8/C8S1_HPS-Application.html">Applications of HPS and HPRS</a></td>
#     <td style="border:none; text-align:left">[Section 8.1.3]<br>Feature enhancement; chroma feature; onset detection; time-scale modification; Violin&ndash;Castanets example</td> 
#     <td style="border:none; text-align:left"><a href="../C8/C8S1_HPS-Application.html">[html]</a></td>      
#     <td style="border:none; text-align:left"><a href="../C8/C8S1_HPS-Application.ipynb">[ipynb]</a></td>    
# </tr>
# 
# <tr text-align="left" style="border:1px solid #C8C8C8">
#     <td style="border:none; text-align:left"><a href="../C8/C8S2_InstantFreqEstimation.html">Instantaneous Frequency Estimation</a></td>
#     <td style="border:none; text-align:left">[Section 8.2.1]<br>Phase wrapping; principle argument; exponential function; phase prediction; instantaneous frequency (IF); polar coordinates; bin offset; visualization of IF values; dependency on hop size; C4 piano example </td> 
#     <td style="border:none; text-align:left"><a href="../C8/C8S2_InstantFreqEstimation.html">[html]</a></td>      
#     <td style="border:none; text-align:left"><a href="../C8/C8S2_InstantFreqEstimation.ipynb">[ipynb]</a></td>    
# </tr>
# 
# <tr text-align="left" style="border:1px solid #C8C8C8">
#     <td style="border:none; text-align:left"><a href="../C8/C8S2_SalienceRepresentation.html">Salience Representation</a></td>
#     <td style="border:none; text-align:left">[Section 8.2.2]<br>Log-frequency spectrogram; instantaneous frequency; binning; harmonic summation; salience; Weber example (Freischütz)</td> 
#     <td style="border:none; text-align:left"><a href="../C8/C8S2_SalienceRepresentation.html">[html]</a></td>      
#     <td style="border:none; text-align:left"><a href="../C8/C8S2_SalienceRepresentation.ipynb">[ipynb]</a></td>    
# </tr>
# 
# <tr text-align="left" style="border:1px solid #C8C8C8">
#     <td style="border:none; text-align:left"><a href="../C8/C8S2_FundFreqTracking.html">Fundamental Frequency Tracking</a></td>
#     <td style="border:none; text-align:left">[Section 8.2.3]<br>Frequency trajectory; sonification; salience representation; continuity constraint; dynamic programming; score-informed constraint; constraint region; Weber example (Freischütz); Bornemark example (Stop Messing With Me)</td> 
#     <td style="border:none; text-align:left"><a href="../C8/C8S2_FundFreqTracking.html">[html]</a></td>      
#     <td style="border:none; text-align:left"><a href="../C8/C8S2_FundFreqTracking.ipynb">[ipynb]</a></td>    
# </tr>
# 
# <tr text-align="left" style="border:1px solid #C8C8C8">
#     <td style="border:none; text-align:left"><a href="../C8/C8S2_MelodyExtractSep.html">Melody Extraction and Separation</a></td>
#     <td style="border:none; text-align:left">[Section 8.2, Section 8.2.3.3]<br>Melody; salience representation; predominant frequency; F0-trajectory; separation; binary mask; harmonics; frequency-dependent tolerance; signal reconstruction; Weber example (Freischütz); Bornemark example (Stop Messing With Me)</td> 
#     <td style="border:none; text-align:left"><a href="../C8/C8S2_MelodyExtractSep.html">[html]</a></td>      
#     <td style="border:none; text-align:left"><a href="../C8/C8S2_MelodyExtractSep.ipynb">[ipynb]</a></td>    
# </tr>
# 
# <tr text-align="left" style="border:1px solid #C8C8C8">
#     <td style="border:none; text-align:left"><a href="../C8/C8S3_NMFbasic.html">Nonnegative Matrix Factorization (NMF)</a></td>
#     <td style="border:none; text-align:left">[Section 8.3.1]<br>Matrix factorization; nonnegative matrix; rank; template vector; activation vector; gradient descent; multiplicative update rule; magnitude spectrogram; Chopin example (Op. 28, No. 4); C-major scale example</td> 
#     <td style="border:none; text-align:left"><a href="../C8/C8S3_NMFbasic.html">[html]</a></td>      
#     <td style="border:none; text-align:left"><a href="../C8/C8S3_NMFbasic.ipynb">[ipynb]</a></td>    
# </tr>
# 
# <tr text-align="left" style="border:1px solid #C8C8C8">
#     <td style="border:none; text-align:left"><a href="../C8/C8S3_NMFSpecFac.html">NMF-Based Spectrogram Factorization</a></td>
#     <td style="border:none; text-align:left">[Section 8.3.2]<br>Spectrogram factorization; score-informed NMF; initialization; template constraints; pitch information; activation constraints; score information; onset model; Chopin example (Op. 28, No. 4)</td> 
#     <td style="border:none; text-align:left"><a href="../C8/C8S3_NMFSpecFac.html">[html]</a></td>      
#     <td style="border:none; text-align:left"><a href="../C8/C8S3_NMFSpecFac.ipynb">[ipynb]</a></td>    
# </tr>
# 
# <tr text-align="left" style="border:1px solid #C8C8C8">
#     <td style="border:none; text-align:left"><a href="../C8/C8S3_NMFAudioDecomp.html">NMF-Based Audio Decomposition</a></td>
#     <td style="border:none; text-align:left">[Section 8.3.3]<br>Score-informed NMF; activation matrix; spectral masking; audio decomposition; audio editing; Chopin example (Op. 28, No. 4)</td> 
#     <td style="border:none; text-align:left"><a href="../C8/C8S3_NMFAudioDecomp.html">[html]</a></td>      
#     <td style="border:none; text-align:left"><a href="../C8/C8S3_NMFAudioDecomp.ipynb">[ipynb]</a></td>    
# </tr>
# </table>

# %% [markdown]
# <table style="border:none">
# <tr style="border:none">
#     <td style="min-width:50px; border:none" bgcolor="white"><a href="../C0/C0.html"><img src="../data/C0_nav.png" style="height:50px" alt="C0"></a></td>
#     <td style="min-width:50px; border:none" bgcolor="white"><a href="../C1/C1.html"><img src="../data/C1_nav.png" style="height:50px" alt="C1"></a></td>
#     <td style="min-width:50px; border:none" bgcolor="white"><a href="../C2/C2.html"><img src="../data/C2_nav.png" style="height:50px" alt="C2"></a></td>
#     <td style="min-width:50px; border:none" bgcolor="white"><a href="../C3/C3.html"><img src="../data/C3_nav.png" style="height:50px" alt="C3"></a></td>
#     <td style="min-width:50px; border:none" bgcolor="white"><a href="../C4/C4.html"><img src="../data/C4_nav.png" style="height:50px" alt="C4"></a></td>
#     <td style="min-width:50px; border:none" bgcolor="white"><a href="../C5/C5.html"><img src="../data/C5_nav.png" style="height:50px" alt="C5"></a></td>
#     <td style="min-width:50px; border:none" bgcolor="white"><a href="../C6/C6.html"><img src="../data/C6_nav.png" style="height:50px" alt="C6"></a></td>
#     <td style="min-width:50px; border:none" bgcolor="white"><a href="../C7/C7.html"><img src="../data/C7_nav.png" style="height:50px" alt="C7"></a></td>
#     <td style="min-width:50px; border:none" bgcolor="white"><a href="../C8/C8.html"><img src="../data/C8_nav.png" style="height:50px" alt="C8"></a></td>
# </tr>
# </table>


# MusicSpeechClassification
Audio classifier for the GTZAN dataset, a small collection of 128 clips of both music and speech. Uses an AdaBoosted Support Vector Machine on the spectrogram representation of the audio files to achieve over 80% classification accuracy. Uses PCA to visualize the data, as well as K-Folds cross validation to ensure network doesn't overfit. 

![Finished Circuit](https://raw.githubusercontent.com/Toback/MusicSpeechClassification/master/Results/SVM_Accuracy_Graph.png)

The PCA projections below demonstrate that the frequency spectrogram representation of the data is preferable to the raw audio, as the clusters are considerable more seperable in the spectrogram graph.
<img src="https://raw.githubusercontent.com/Toback/MusicSpeechClassification/master/Results/Audio_File_PCA.png" width="425"/> <img src="https://raw.githubusercontent.com/Toback/MusicSpeechClassification/master/Results/Spectrogram_PCA.png" width="425"/> 

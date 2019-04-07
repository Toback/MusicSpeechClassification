# MusicSpeechClassification
For this project I developed two audio classifiers for the GTZAN dataset, a small collection of 128 clips of both music and speech. The goal was to create a model which could predict whether a given audio clip was music with singing in it or whether it was human speech. Below are examples taken from the test set.

Music

Speech

First, I used an AdaBoosted Support Vector Machine on the spectrogram representation of the audio files to achieve over 80% classification accuracy. Second, I used a Convolutional Neural Network and was able to produce over 95% accuracy on my test set. I also used PCA to visualize the data, as well as K-Folds cross validation to ensure the network didn't overfit. 

![Finished Circuit](https://raw.githubusercontent.com/Toback/MusicSpeechClassification/master/Results/SVM_Accuracy_Graph.png)

The PCA projections below demonstrate that the frequency spectrogram representation of the data is preferable to the raw audio, as the clusters are considerable more seperable in the spectrogram graph.
<img src="https://raw.githubusercontent.com/Toback/MusicSpeechClassification/master/Results/Audio_File_PCA.png" width="425"/> <img src="https://raw.githubusercontent.com/Toback/MusicSpeechClassification/master/Results/Spectrogram_PCA.png" width="425"/> 

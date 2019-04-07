# MusicSpeechClassification
For this project I developed two audio classifiers for the GTZAN dataset, a small collection of 128 clips of both music and speech. The goal was to create models which could predict whether a given audio clip was either music with singing in it or plain human speech. Below are examples taken from the test set.

[Music Example](GTZAN%20Examples/bagpipe.wav)

[Speech Example](GTZAN%20Examples/comedy.wav)

First, I used an AdaBoosted Support Vector Machine with a linear kernel and achieved over 80% classification accuracy. Then I used a three-layer Convolutional Neural Network and was able to produce over 95% accuracy on my test set. I also used PCA to visualize the data, as well as [K-Folds cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation) to ensure the network didn't overfit.

The SVM's accuracy graph is below, with the x-axis being how many folds were used in the cross-validation. The accuracy at a particular fold was taken to be the average of all the models at that fold. 

![SVM Accuracy](Results/SVM_Accuracy_Graph.png)

The CNN's accuracy graph is below and was created with TensorBoard. The transparent line is the true accuracy of the model, while the bold line is the same line but smoothed out. 

![CNN Accuracy](Results/CNN_Accuracy_Graph.png)

For both models I first changed the data representation from raw audio files to their corresponding frequency spectrograms which can be seen below. 

<img src=Results/Raw_Audio.png width="425"/> <img src=Results/Spectrogram.png width="425"/> 

The PCA projections below motivate using the frequency spectrogram representation of the audio, as the clusters are considerably more seperable in the spectrogram PCA graph.

<img src="https://raw.githubusercontent.com/Toback/MusicSpeechClassification/master/Results/Audio_File_PCA.png" width="425"/> <img src="https://raw.githubusercontent.com/Toback/MusicSpeechClassification/master/Results/Spectrogram_PCA.png" width="425"/> 

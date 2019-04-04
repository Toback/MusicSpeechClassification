##################################################################################
# Music Speech Classification                                                    #
# by Asher Toback                                                                #
# March 2019                                                                     #
# https://github.com/Toback/MusicSpeechClassification                            #
# ------------------------------------------------------------------------------ #
# Audio classifier for the GTZAN dataset, a small collection of 128 clips of     #
# both music and speech. Uses an AdaBoosted Support Vector Machine on the        #
# spectrogram representation of the audio files to achieve over 80%              #
# classification accuracy. Uses Principle Component Analysis (PCA) to visualize  #
# the data, as well as K-Folds cross validation to ensure network doesn't        #
# overfit.                                                                       #
#                                                                                #
# The GTZAN Music Speech dataset can be downloaded at the link below. To run     #
# this code simply update the 'speech_folder_path' and 'music_folder_path' to    #
# wherever you downloaded them onto your machine.                                #
# http://marsyas.info/downloads/datasets.html                                    #
#                                                                                #
##################################################################################

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import librosa.display
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA

# Minimum K in K-Fold Cross-Validation used in SVM experiments
MINFOLDS = 6

# Maximum K in K-Fold Cross-Validation used in SVM experiments
MAXFOLDS = 11

# Sample rate of audio files from GTZAN. Used to create spectrograms
SAMPLERATE = 22050

# Number of estimators used when AdaBoosting the linear SVM
NUMADABOOSTESTIMATORS = 500

# Controls whether PCA is performed on raw audio files or spectrograms
SPECTROGRAMPCA = False
speech_folder_path = '/Users/asher/Desktop/ML_Datasets/music_speech/speech_wav'
music_folder_path = '/Users/asher/Desktop/ML_Datasets/music_speech/music_wav'

def read_data(folder_path):
    """Reads .wav files from folder_path into an array"""
    data = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        clip, sr = librosa.load(filepath)
        data.append(clip)
    return np.array(data)


def freq_conversion(time_data, sample_rate):
    """Converts time series data into a spectrogram of frequencies.
    Specifically, the Mel-Spectrogram"""
    freq_data = []
    for i in range(len(time_data)):
        freq_data.append(
            np.array(
                librosa.feature.melspectrogram(y=time_data[i], sr=sample_rate))
            .flatten())
    return np.array(freq_data)


def unison_shuffled_copies(a, b):
    """Taken from stack overflow. Identically permutes three arrays"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def confusion_matrix(ans, predicts):
    """Constructs confusion matrix with actual answers and
    predicted answers, which are provided as input"""
    confusion_answer = pd.Series(ans, name='Answer')
    confusion_predicted = pd.Series(predicts, name='Predicted')
    return pd.crosstab(confusion_answer, confusion_predicted)


def prediction_stats(ans, predicts):
    """Calculates Accuracy of prediction against ground truth labels"""
    total_correct = 0
    for i in range(len(predicts)):
        if ans[i] == predicts[i]:
            total_correct = total_correct + 1
    return total_correct / float(len(predicts))


def main():
    # Read all .wav files, convert them to Mel-Spectrograms, generate
    # ground truth labels for data, and finally normalize the data.
    print("\nLoading .wav files")
    speech_data = read_data(speech_folder_path)
    music_data = read_data(music_folder_path)
    print("Finished Loading")
    t_data = np.concatenate((speech_data, music_data))
    print("Converting to Frequency Domain")
    f_data = freq_conversion(t_data, SAMPLERATE)
    print("Finished Conversion ")
    labels = np.append(np.full(len(speech_data), 0),
                       np.full(len(music_data), 1))
    f_data = preprocessing.normalize(f_data)
    t_data = preprocessing.normalize(t_data)


    ########################################################################
    #                                                                      #
    #                      PCA Data Visualization                          #
    #                                                                      #
    ########################################################################
    # Project input data down to its two most separable dimensions found by
    # Principle Component Analysis. Plot result.
    print("\n--------------------------")
    print("PRINCIPLE COMPONENT ANALYSIS")
    print("--------------------------")
    print("\nComputing PCA")
    pca = PCA(n_components=2)
    if SPECTROGRAMPCA:
        pca = pca.fit_transform(f_data)
    else:
        pca = pca.fit_transform(t_data)
    speech_pca_x = [i[0] for i in pca[len(speech_data):]]
    speech_pca_y = [i[1] for i in pca[len(speech_data):]]
    music_pca_x = [i[0] for i in pca[:len(music_data)]]
    music_pca_y = [i[1] for i in pca[:len(music_data)]]

    plt.scatter(speech_pca_x, speech_pca_y, color='red')
    plt.scatter(music_pca_x, music_pca_y, color='blue')
    red_patch = mpatches.Patch(color='red', label='Speech')
    blue_patch = mpatches.Patch(color='blue', label='Music')
    plt.legend(handles=[red_patch, blue_patch])
    plt.title('Spectrogram PCA')
    plt.xlabel('Principle Component 1')
    plt.ylabel('Principle Component 2')
    plt.show()
    print("Finished PCA")

    #########################################################################
    #                                                                       #
    #                  AdaBoosted Support Vector Machine                    #
    #                                                                       #
    #########################################################################
    # This algorithm uses a support vector machine trained on frequency data
    # to make its predictions. A linear kernel is used for the SVM, and it is
    # AdaBoosted to increase its performance by around 3%.
    print("\n--------------------------")
    print("ADABOOST")
    print("--------------------------")

    # Arrays for printing out average accuracy and confusion matrices
    # over the course of an epoch
    time = []
    accuracy = []
    cm = []

    # Perform K-Folds Cross-Validation on the dataset, where 'K' is
    # controlled by the range of the 'for' loop
    for i in range(MINFOLDS, MAXFOLDS+1, 1):
        # Prepare for training epoch. Set the number of folds, zero out
        # epoch arrays for accuracy and confusion matrices, and shuffle
        # frequency data.
        print("\n\nNUM FOLDS : " + str(i) + "\n")
        kf = KFold(n_splits=i)
        episode_accuracy = np.full((kf.get_n_splits(f_data)), 0, dtype=float)
        episode_cm = np.full((kf.get_n_splits(f_data), 2, 2), 0, dtype=float)
        f_data, labels = unison_shuffled_copies(f_data, labels)
        j = 0
        for train_index, test_index in kf.split(f_data):
            # Pull out the training and test data specified by the indices
            # generated by K-Folds. Data Augmentation such as doubling or
            # reversing could be done here.
            train_f_data, test_f_data = f_data[train_index], f_data[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]

            # Create, fit, and predict with an AdaBoosted Support Vector Machine
            # which uses a linear kernel.
            flin_clf = AdaBoostClassifier(svm.LinearSVC(),
                                          n_estimators=NUMADABOOSTESTIMATORS,
                                          learning_rate=1.0, algorithm='SAMME')
            print("\tFitting")
            flin_clf.fit(train_f_data, train_labels)
            print("\tPredicting")
            flin_test_predictions = flin_clf.predict(test_f_data)

            # Check accuracy and confusion matrix of the model and store for future
            # analysis
            current_accuracy = prediction_stats(test_labels, flin_test_predictions)
            current_cm = confusion_matrix(test_labels, flin_test_predictions)
            episode_accuracy[j] = current_accuracy
            episode_cm[j] = current_cm
            print("\tCurrent Accuracy: " + str(episode_accuracy[j]))
            j = j + 1

        # Compute average performance over the K-Folding epoch.
        print("\nAverage Accuracy: " + str(np.mean(np.array(episode_accuracy))))
        print("Average Confusion Matrix:")
        print(str(np.mean(np.array(episode_cm), axis=0)))
        accuracy.append(np.mean(np.array(episode_accuracy)))
        cm.append(np.mean(np.array(episode_cm), axis=0))
        time.append(i)

    # Print out results of training and testing
    best_accuracy = np.argmax(np.array(accuracy))
    print("\n\n CONCLUSION\n")
    print("The best accuracy is " + str(np.array(accuracy)[best_accuracy]))
    print("It has confusion matrix")
    print(np.array(cm)[best_accuracy])
    print("And it occurred when the K-Folds split was "
          + str(MINFOLDS + best_accuracy))
    plt.title('SVM Classification Accuracy at different K-Fold Values')
    plt.xlabel('Number of Folds used in Cross-Validation')
    plt.ylabel('Accuracy %')
    plt.plot(time, accuracy)
    plt.show()


if __name__ == "__main__":
    main()

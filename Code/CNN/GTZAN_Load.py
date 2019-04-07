#############################################################################################
# GTZAN_Load                                                                                #
# by Asher Toback                                                                           #
# April 2019                                                                                #
# https://github.com/Toback/MusicSpeechClassification                                       #
# ----------------------------------------------------------------------------------------- #
# Class which can be used to load the GTZAN dataset for use in machine learning algorithms. #
# Training and test data, along with the corresponding labels, for either the raw audio     #
# files or spectrogram images can be loaded from a provided raw_audio_path. Images produced #
# are set to be 80 x 80 but this can be changed by adjusting the IMAGESIZE constant.        #
#                                                                                           #
#############################################################################################

import numpy as np
import os
import librosa.display
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Constants
SAMPLERATE = 22050
IMAGESIZE= 80

# Pyplot settings for an IMAGESIZE x IMAGESIZE pixel image, no border, no axis names, and no
# title. This size image allows for enough information to classify accurately while still
# being small enough to allow for a model to be trained quickly.
plt.figure(figsize=(IMAGESIZE / 100.0, IMAGESIZE / 100.0))
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.axis('tight')
plt.axis('off')
plt.box(False)


class GTZAN:
    """Class which reads, processes, saves, and returns the GTZAN dataset in two
    different formats: raw audio, or 80 x 80 images. If load_images is called and
    no images are found at the provided 'saved_images_path', then the raw audio
    is loaded and the images are created and saved at the path provided."""
    def __init__(self, raw_audio_path, num_train_examples, saved_images_path=None):
        self.raw_audio_path = raw_audio_path
        self.saved_images_path = saved_images_path
        self.num_train_examples = num_train_examples
        return

    def load_images(self):
        """Reads spectrogram images from 'self.saved_images_path', normalizes the images,
        and then returns the training and test data, along with the corresponding labels."""
        def load_image_directory(dir_path, type):
            """Loads every spectrogram image in a directory into an array"""
            imgs = np.zeros(shape=(len(os.listdir(dir_path)), IMAGESIZE, IMAGESIZE))
            for i in range(len(imgs)):
                img = Image.open(dir_path + type + '{0}.png'.format(i))
                imgs[i] = np.array(img.convert('L'))
            return imgs
        # Check if any of the spectrogram image directories are empty. If they are then
        # this means we haven't created the images yet and thus we need to.
        subdir_empty = [(len(os.listdir(x[0])) == 0) for x in os.walk(self.saved_images_path)]
        if any(subdir_empty):
            print("Images not found, creating images.")
            self.create_spectrogram_images()
        print("\nLoading images")
        speech_train_folder_path = os.path.join(self.saved_images_path, 'speech_train/')
        speech_test_folder_path = os.path.join(self.saved_images_path, 'speech_test/')
        music_train_folder_path = os.path.join(self.saved_images_path, 'music_train/')
        music_test_folder_path = os.path.join(self.saved_images_path, 'music_test/')
        speech_train = load_image_directory(speech_train_folder_path, 'speech')
        speech_test = load_image_directory(speech_test_folder_path, 'speech')
        music_train = load_image_directory(music_train_folder_path, 'music')
        music_test = load_image_directory(music_test_folder_path, 'music')
        print("Finished Loading")
        training_data = np.concatenate((speech_train, music_train))
        testing_data = np.concatenate((speech_test, music_test))
        training_labels = np.append(np.full(len(speech_train), 0),
                                    np.full(len(music_train), 1))
        testing_labels = np.append(np.full(len(speech_test), 0),
                                   np.full(len(music_test), 1))
        return ((self.unison_shuffled_copies(training_data, training_labels)),
                (self.unison_shuffled_copies(testing_data, testing_labels)))

    def load_raw_audio(self):
        """Reads raw audio files and randomly partitions them into training and testing
        data arrays. This data along with the corresponding labels are returned"""
        speech_folder_path = os.path.join(self.raw_audio_path, 'speech_wav')
        music_folder_path = os.path.join(self.raw_audio_path, 'music_wav')
        speech_data = self.read_data(speech_folder_path)
        music_data = self.read_data(music_folder_path)
        speech_train, speech_test = self.split(speech_data, self.num_train_examples)
        music_train, music_test = self.split(music_data, self.num_train_examples)
        training_data = np.concatenate((speech_train, music_train))
        testing_data = np.concatenate((speech_test, music_test))
        training_labels = np.append(np.full(len(speech_train), 0),
                                    np.full(len(music_train), 1))
        testing_labels = np.append(np.full(len(speech_test), 0),
                                   np.full(len(music_test), 1))
        return ((self.unison_shuffled_copies(training_data, training_labels)),
                (self.unison_shuffled_copies(testing_data, testing_labels)))

    def create_spectrogram_images(self):
        """Reads raw audio files, randomly partitions them into training and test data,
        converts them into IMAGESIZE x IMAGESIZE Mel-Spectrograms, and then saves them
        to 'self.saved_images_path'. """
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
        print("\nLoading .wav files")
        speech_folder_path = os.path.join(self.raw_audio_path, 'speech_wav')
        music_folder_path = os.path.join(self.raw_audio_path, 'music_wav')
        speech_data = self.read_data(speech_folder_path)
        music_data = self.read_data(music_folder_path)
        speech_train, speech_test = self.split(speech_data, self.num_train_examples)
        music_train, music_test = self.split(music_data, self.num_train_examples)
        print("Finished Loading")
        print("Converting to Frequency Domain and Normalizing")
        speech_train_f = freq_conversion(speech_train, SAMPLERATE)
        music_train_f = freq_conversion(music_train, SAMPLERATE)
        speech_test_f = freq_conversion(speech_test, SAMPLERATE)
        music_test_f = freq_conversion(music_test, SAMPLERATE)
        speech_train_f = preprocessing.normalize(speech_train_f)
        music_train_f = preprocessing.normalize(music_train_f)
        speech_test_f = preprocessing.normalize(speech_test_f)
        music_test_f = preprocessing.normalize(music_test_f)
        print("Finished Conversion\nCreating Spectrograms")
        self.save_spectrograms(speech_train_f, "speech",
                               self.saved_images_path + '/speech_train')
        self.save_spectrograms(music_train_f, "music",
                               self.saved_images_path + '/music_train')
        self.save_spectrograms(speech_test_f, "speech",
                               self.saved_images_path + '/speech_test')
        self.save_spectrograms(music_test_f, "music",
                               self.saved_images_path + '/music_test')
        print("Saving Images")
        return


    def read_data(self, folder_path):
        """Reads .wav files from folder_path into an array"""
        data = []
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            clip, sr = librosa.load(filepath)
            data.append(clip)
        return np.array(data)

    def split(self, data, num_split):
        """Randomly partition data into two arrays, one of size 'num_split' and the other
        the remaining elements."""
        p = np.random.permutation(len(data))
        p_data = data[p]
        return p_data[:num_split], p_data[num_split:]

    def unison_shuffled_copies(self, a, b):
        """Taken from stack overflow. Identically permutes three arrays"""
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def save_spectrograms(self, signals, name, path):
        """Create gray-scale Mel-Spectrogram image of raw audio signal, then save image
        to 'path'. """
        for i in range(len(signals)):
            spect = librosa.feature.melspectrogram(y=signals[i], sr=SAMPLERATE)
            librosa.display.specshow(librosa.power_to_db(spect, ref=np.max),
                                     cmap='gray_r')
            plt.savefig(path+'/'+name+str(i))


def main():
    ds = GTZAN(raw_audio_path="/Users/asher/Desktop/ML_Datasets/music_speech/",
               saved_images_path="/Users/asher/PycharmProjects/FirstSteps/freq_images/",
               num_train_examples=50)
    ds.load_images()


if __name__ == "__main__":
    main()


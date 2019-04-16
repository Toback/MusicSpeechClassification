################################################################################
# Music Speech Classification -- CNN Implementation                            #
# by Asher Toback                                                              #
# April 2019                                                                   #
# https://github.com/Toback/MusicSpeechClassification                          #
# ---------------------------------------------------------------------------- #
# Audio classifier for the GTZAN dataset, a small collection of 128 clips of   #
# both music and speech. Uses a two-layer Convolutional Neural Network on the  #
# spectrogram representation of the audio files to achieve over 95%            #
# classification accuracy. Images used are grey-scale and 80 x 80 pixels in    #
# size. For details of the CNN model's architecture look to the 'cnn_model_fn' #
# comments below.                                                              #
#                                                                              #
# The GTZAN Music Speech dataset can be downloaded at the link below. To run   #
# this code simply update the 'raw_audio_path' to wherever you downloaded them #
# onto your machine.                                                           #
# http://marsyas.info/downloads/datasets.html                                  #
#                                                                              #
################################################################################
from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import GTZAN_Load as MusicSpeechDataset

# Training Hyper Parameters
ETA = 1e-2
NUM_TRAINING_STEPS = 1500
STEPS_UNTIL_LOG_RESULTS = 50
BATCH_SIZE = 10
DROPOUT = 0.4

# Network Constants and Paths
IMAGESIZE = 80
NUM_CLASSES = 2
raw_audio_path="/Users/asher/Desktop/ML_Datasets/music_speech/"
saved_images_path="/Users/asher/PycharmProjects/FirstSteps/freq_images/"


def cnn_model_fn(features, labels, mode):
    """Implementation of the CNN used for classification of the GTZAN dataset.
    The first layer has 32 filters, stride of 5, and uses ReLu activations
    and is followed by a max-pooling layer with filters of width 2. The second
    convolutional layer is identical but has 64 filters and is similarly
    followed by a max-pooling layer. There's then one fully connected layer of
    1024 neurons, which is followed by a softmax layer used for the final
    classification. Stochastic Gradient Descent and cross-entropy loss are
    used to train the model, while dropout is used to prevent overfitting."""
    input_layer = tf.reshape(features["x"], [-1, IMAGESIZE, IMAGESIZE, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=2)

    conv2 = tf.layers.conv2d(
        inputs= pool1,
        filters= 64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 20*20*64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Dropout is only applied to the network during TRAINING.
    dropout = tf.layers.dropout(
        inputs=dense, rate=DROPOUT, training=(mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(inputs=dropout, units=NUM_CLASSES)

    # Dictionary used to hold the outputs/predictions of the model.
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # If the model was called from tf.estimator.Estimator().evaluate then
    # it will be in prediction mode and thus the outputs are returned.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # If the model was called from tf.estimator.Estimator().train the use
    # the loss calculated above to train the model by SGD.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=ETA)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # If the model is neither being TRAINED or used to PREDICT then it's being
    # being probed to see how it's classification accuracy is progressing on the
    # training set. This happens every 50 training steps and the data will be
    # stored in the 'model_dir'. The graphs of these statisticscan be viewed
    # used tensorboard, where log_dir='model_dir'/eval
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    # Load saved images from saved_images path. If not all the images
    # are found then go to the source of the raw audio files and
    # create them, using 'num_train_examples' as the number of
    # training examples.
    ds = MusicSpeechDataset.GTZAN(raw_audio_path=raw_audio_path,
                                  saved_images_path=saved_images_path,
                                  num_train_examples=50)
    ((train_data, train_labels), (test_data, test_labels)) = ds.load_images()

    # Scale gray-scale image pixel values between 0 and 1 and make labels
    # readable to the CNN.
    train_data = train_data / 255.0
    test_data = test_data / 255.0
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)

    # Instantiate the gtzan classifier as the CNN model described above
    # and save all summary statstics to 'model_dir'
    gtzan_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="/Users/asher/PycharmProjects/FirstSteps/gtzan_cnn_log"
    )

    # When training use the following training data and labels to train
    # against, shuffle the order of the data, and batch the data. Set no
    # limit for the number of epochs allowable to train over.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True
    )

    # Only test for one epoch.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )

    # Train the CNN for STEPS_UNTIL_LOG_RESULTS steps, and then evaluate
    # the model in order to create summary statistics to view later in
    # tensorboard. Do this until NUM_TRAINING_STEPS steps have elapsed.
    for i in range(int(round(NUM_TRAINING_STEPS/STEPS_UNTIL_LOG_RESULTS))):
        gtzan_classifier.train(
            input_fn=train_input_fn,
            steps=STEPS_UNTIL_LOG_RESULTS
        )
        test_results = gtzan_classifier.evaluate(input_fn=test_input_fn)
        print(test_results)


if __name__ == "__main__":
    main()


"""
This is converted from tensorflow simple audio recognition tutorial: https://www.tensorflow.org/tutorials/audio/simple_audio
"""

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

AUTOTUNE = tf.data.AUTOTUNE


class SpeechCommandModel():

    def preprocess_dataset(self, files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        # show_wav_ds(output_ds)
        # show_spectrogram(output_ds)
        output_ds = output_ds.map(
            lambda audio, label: get_spectrogram_and_label_id(audio, label, self.commands), num_parallel_calls=AUTOTUNE)
        # show_all_spectrogram(spectrogram_ds, commands)
        def spectrogram_preprocess(spectrogram, label_id):
            spectrogram = preprocessing.Resizing(32, 32, interpolation='area')(spectrogram)
            spectrogram = preprocessing.Normalization(mean=0.0, variance=1.0)(spectrogram) # [-1,1] conceptually
            # spectrogram = preprocessing.Rescaling(0.5, offset=1.0)(spectrogram) # [0,1] conceptually
            return spectrogram, label_id
        output_ds = output_ds.map(spectrogram_preprocess)
        return output_ds

    def load_mini_speech_commands_dataset(self, data_dir):
        data_dir = pathlib.Path(data_dir)
        # Set seed for experiment reproducibility
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.commands = np.array(tf.io.gfile.listdir(str(data_dir)))
        self.commands = self.commands[self.commands != 'README.md']
        print('Commands:', self.commands)

        filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
        filenames = tf.random.shuffle(filenames)
        num_samples = len(filenames)
        print('Number of total examples:', num_samples)
        print('Number of examples per label:',
              len(tf.io.gfile.listdir(str(data_dir / self.commands[0]))))
        print('Example file tensor:', filenames[0])

        train_files = filenames[:6400]
        val_files = filenames[6400: 6400 + 800]
        test_files = filenames[-800:]

        print('Training set size', len(train_files))
        print('Validation set size', len(val_files))
        print('Test set size', len(test_files))

        self.train_ds = self.preprocess_dataset(train_files)
        self.val_ds = self.preprocess_dataset(val_files)
        self.test_ds = self.preprocess_dataset(test_files)

    def build(self, input_shape, num_labels):
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_labels),
        ])
        self.model.summary()
        return self.model

    def train(self, batch_size=64, EPOCHS=10):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        self.train_ds = self.train_ds.batch(batch_size)
        self.val_ds = self.val_ds.batch(batch_size)
        self.train_ds = self.train_ds.cache().prefetch(AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(AUTOTUNE)


        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=EPOCHS,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        )
        return history

    def test(self):
        test_audio = []
        test_labels = []

        for audio, label in self.test_ds:
            test_audio.append(audio.numpy())
            test_labels.append(label.numpy())

        test_audio = np.array(test_audio)
        test_labels = np.array(test_labels)

        y_pred = np.argmax(self.model.predict(test_audio), axis=1)
        y_true = test_labels

        test_acc = sum(y_pred == y_true) / len(y_true)
        print(f'Test set accuracy: {test_acc:.0%}')

    def save(self,fp):
        self.model.save(fp)

    def load_model(self, fp):
        self.model = models.load_model(filepath=fp)
        return self.model


def show_all_spectrogram(spectrogram_ds, commands):
    rows = 3
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
        ax.set_title(commands[label_id.numpy()])
        ax.axis('off')

    plt.show()


def get_spectrogram_and_label_id(audio, label, commands):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


def show_wav_ds(waveform_ds):
    rows = 3
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
    for i, (audio, label) in enumerate(waveform_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(audio.numpy())
        ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        label = label.numpy().decode('utf-8')
        ax.set_title(label)

    plt.show()


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)

    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns). An epsilon is added to avoid log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def show_spectrogram(waveform_ds):
    for waveform, label in waveform_ds.take(1):
        label = label.numpy().decode('utf-8')
        spectrogram = get_spectrogram(waveform)

        fig, axes = plt.subplots(2, figsize=(12, 8))
        timescale = np.arange(waveform.shape[0])
        axes[0].plot(timescale, waveform.numpy())
        axes[0].set_title('Waveform')
        axes[0].set_xlim([0, 16000])
        plot_spectrogram(spectrogram.numpy(), axes[1])
        axes[1].set_title('Spectrogram')
        plt.show()


if __name__ == "__main__":
    data_dir = "data/0_data/mini_speech_commands"
    mModel = SpeechCommandModel()
    mModel.load_mini_speech_commands_dataset(data_dir)
    for spectrogram, _ in mModel.train_ds.take(1):
        input_shape = spectrogram.shape
    print('Input shape:', input_shape)
    num_labels = len(mModel.commands)
    print('Output Class:', num_labels)
    mModel.build(input_shape=input_shape,num_labels=num_labels)
    mModel.train()
    mModel.test()
    model_output_dir = "model/SavedModel/SpeechCommandModel/"
    mModel.save(model_output_dir)

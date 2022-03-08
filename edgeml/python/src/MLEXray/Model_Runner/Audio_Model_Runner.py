import os
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

from MLEXray.EdgeMLMonitor import EdgeMLMonitor
from MLEXray.Model_Runner.Model_Runner import Model_Runner
from MLEXray.Utils.params import ModelName
from MLEXray.Utils.tf_utils import KerasModel_preprocessing
import cv2
import tensorflow_hub as hub

from IPython import display
import matplotlib.pyplot as plt

import pandas as pd
import tensorflow_io as tfio
import scipy
from scipy.io import wavfile

class Audio_Model_Runner(Model_Runner):

    def ensure_sample_rate(self, original_sample_rate, waveform,
                           desired_sample_rate=16000):
        """Resample waveform if required."""
        if original_sample_rate != desired_sample_rate:
            desired_length = int(round(float(len(waveform)) /
                                       original_sample_rate * desired_sample_rate))
            waveform = scipy.signal.resample(waveform, desired_length)
        return desired_sample_rate, waveform

    def run_one_audio(self, audio_path):
        """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
        file_contents = tf.io.read_file(audio_path)
        wav, sample_rate = tf.audio.decode_wav(
            file_contents,
            desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        # wav = np.expand_dims(wav, axis=0)
        # print(wav.shape)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        audio = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)


        # sample_rate, wav = wavfile.read(audio_path, 'rb')
        # sample_rate, wav = self.ensure_sample_rate(sample_rate, wav)
        # normalization
        # audio = wav / tf.int16.max # already normalized
        audio = tf.cast(audio, tf.float32)
        print(audio.shape)


        # model = hub.load('https://tfhub.dev/google/yamnet/1')
        # scores, embeddings, spectrogram = model(audio)


        # self.mMonitor.onInferenceStart()
        result = self.invoke_model(audio) # [scores, embeddings, spectrogram]
        # self.mMonitor.onInferenceStop(scores, raw_input=wav)

        return result
        # class_scores = tf.reduce_mean(scores, axis=0)
        # top_class = tf.argmax(class_scores)
        # inferred_class = class_names[top_class]
        #
        # print(f'The main sound is: {inferred_class}')
        # print(f'The embeddings shape: {embeddings.shape}')

    # @staticmethod
    # def preprocess_tfdataset_audio(data, model_name, input_size=224):
    #     def preprocess_dataset(files):
    #         files_ds = tf.data.Dataset.from_tensor_slices(files)
    #         output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    #         output_ds = output_ds.map(
    #             get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    #         return output_ds
    #
    #
    #     audio = data['audio']
    #     print(audio)
    #     audio = tf.cast(audio, tf.float32)
    #     audio = KerasModel_preprocessing(audio, model_name=model_name)
    #     audio = tf.image.resize(audio, [input_size, input_size], method=tf.image.ResizeMethod.AREA)
    #     print(audio)
    #     label = tf.one_hot(data['label'], depth=1000)
    #
    #     data['image'] = audio
    #     data['label'] = label
    #     return data['image'], data['label']

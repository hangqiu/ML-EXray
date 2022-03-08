import os
import pathlib

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from MLEXray.EdgeMLMonitor import EdgeMLMonitor
from MLEXray.Model_Runner.Audio_Model_Runner import Audio_Model_Runner
from MLEXray.Model_Runner.Image_Model_Runner import Image_Model_Runner
from MLEXray.Model_Runner.Model_Runner import Model_Runner
from MLEXray.Model_Trainer.SpeechCommandModel import SpeechCommandModel, get_waveform_and_label, \
    get_spectrogram_and_label_id
from MLEXray.Utils.params import ModelName
from MLEXray.Utils.tf_utils import KerasModel_preprocessing
import cv2
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental import preprocessing

class Keras_Audio_Model_Runner(Audio_Model_Runner):
    embed_layer_name = None
    def __init__(self, model_name, eval=False):
        if model_name == ModelName.SpeechCommandModel:
            mModel=SpeechCommandModel()
            self.core = mModel.load_model("model/SavedModel/SpeechCommandModel")
        else:
            raise ValueError(f"Model name {self.model_name} not supported yet!")

        if self.embed_layer_name is not None:
            self.embed_layer = self.core.get_layer(self.embed_layer_name).output
            print(self.embed_layer)
        # self.model = tf.keras.Model(inputs=[i], outputs=[x, self.embed_layer])
        if eval or self.embed_layer_name is None:
            self.model = tf.keras.Model(inputs={'input': self.core.input},
                                        outputs={'predictions': self.core.output})
        else:
            self.model = tf.keras.Model(inputs={'input': self.core.input},
                                    outputs={'predictions': self.core.output,
                                             'embedding': self.embed_layer,
                                             'input': self.core.input})
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        self.model.summary()
        print(f"# Layers: {len(self.model.layers)}")
        self.input_size = self.model.input_shape['input'][-2]

        super().__init__(model_name)

    def invoke_model(self, input):
        result = self.model.predict(input)
        return result


    def run_one_audio_folder(self, data_dir):

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

        # filenames = filenames[-800:]
        # num_samples = len(test_files)
        print('Number of total examples:', num_samples)

        files_ds = tf.data.Dataset.from_tensor_slices(filenames)
        output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        output_ds = output_ds.map(
            lambda audio, label: get_spectrogram_and_label_id(audio, label, self.commands), num_parallel_calls=tf.data.AUTOTUNE)

        def spectrogram_preprocess(spectrogram, label_id):
            # spectrogram = preprocessing.Resizing(32, 32, interpolation='area')(spectrogram)
            spectrogram = preprocessing.Resizing(32, 32, interpolation='bilinear')(spectrogram)
            spectrogram = preprocessing.Normalization(mean=0.0, variance=1.0)(spectrogram) # [-1,1] conceptually
            # spectrogram = preprocessing.Rescaling(0.5, offset=1.0)(spectrogram) # [0,1] conceptually
            return spectrogram, label_id
        output_ds = output_ds.map(spectrogram_preprocess)

        test_audio = []
        test_labels = []

        for audio, label in output_ds:
            test_audio.append(audio.numpy())
            test_labels.append(label.numpy())

        test_audio = np.array(test_audio)
        test_labels = np.array(test_labels)
        y_pred = self.model.predict(test_audio)['predictions']
        # print(y_pred.shape)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = test_labels

        test_acc = sum(y_pred == y_true) / len(y_true)
        print(f'Test set accuracy: {test_acc:.2%}')
        return test_acc



if __name__ == '__main__':
    # run data trace
    trace_name = "mini_speech_commands"
    data_path = f"data/0_data/{trace_name}/"


    model_name = ModelName.SpeechCommandModel

    runner = Keras_Audio_Model_Runner(model_name=model_name)
    log_path = f"data/trace_{trace_name}/{model_name}/1_cloud/{trace_name}/"

    runner.run_one_audio_folder(data_path)

    # # evaluate on public dataset
    # runner = Model_Runner('MobileNetV2', eval=True)
    # # dataset = tfds.load('imagenet_v2', split='test', shuffle_files=True)
    # dataset = tfds.load('imagenet2012', split='validation', shuffle_files=True)
    # # dataset = tfds.load('mnist', split='test', shuffle_files=True)
    # runner.evaluate_on_dataset(dataset, batch_size=128)
    # # runner.train_on_dataset(dataset)

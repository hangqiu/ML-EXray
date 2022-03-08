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
from MLEXray.Model_Trainer.TextClassifierModel import TextClassifierModel
from MLEXray.Utils.params import ModelName
from MLEXray.Utils.tf_utils import KerasModel_preprocessing
import cv2
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental import preprocessing

class Keras_Text_Model_Runner(Model_Runner):
    embed_layer_name = None
    def __init__(self, model_name, eval=False):
        if model_name == ModelName.TextClassifierModel:
            self.mModel=TextClassifierModel()
            self.core = self.mModel.load_model("model/SavedModel/TextClassificationModel")
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
        self.input_size = self.model.input_shape['input'][0]

        super().__init__(model_name)

    def invoke_model(self, input):
        result = self.model.predict(input)
        return result


    def run_imdb(self):
        train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"],
                                          batch_size=-1, as_supervised=True)

        self.train_examples, self.train_labels = tfds.as_numpy(train_data)
        self.test_examples, self.test_labels = tfds.as_numpy(test_data)

        # print(self.train_examples[0])
        # self.train_examples = np.array([str(x).lower() for x in self.train_examples])
        # print(self.train_examples[0])

        print("Training entries: {}, test entries: {}".format(len(self.train_examples), len(self.test_examples)))
        results = self.core.evaluate(self.test_examples, self.test_labels)
        return results



if __name__ == '__main__':
    model_name = ModelName.TextClassifierModel
    mRunner = Keras_Text_Model_Runner(model_name)
    mRunner.run_imdb()


import keras.models
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


class TextClassifierModel():
    def load_IMDB_tfds(self):
        train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"],
                                          batch_size=-1, as_supervised=True)

        self.train_examples, self.train_labels = tfds.as_numpy(train_data)
        self.test_examples, self.test_labels = tfds.as_numpy(test_data)

        print(self.train_examples[0])
        self.train_examples = np.array([str(x).lower() for x in self.train_examples])
        print(self.train_examples[0])

        print("Training entries: {}, test entries: {}".format(len(self.train_examples), len(self.test_examples)))

        self.x_val = self.train_examples[:10000]
        self.partial_x_train = self.train_examples[10000:]

        self.y_val = self.train_labels[:10000]
        self.partial_y_train = self.train_labels[10000:]

    def build(self):
        hub_model = "https://tfhub.dev/google/nnlm-en-dim50/2"
        # hub_model = "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2"
        # hub_model = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2"
        hub_layer = hub.KerasLayer(hub_model, input_shape=[], dtype=tf.string, trainable=True)
        self.model = tf.keras.Sequential()
        self.model.add(hub_layer)
        self.model.add(tf.keras.layers.Dense(16, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1))

        self.model.summary()
        return self.model

    def train(self):
        self.model.compile(optimizer='adam',
                      loss=tf.losses.BinaryCrossentropy(from_logits=True),
                      metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])

        history = self.model.fit(self.partial_x_train,
                            self.partial_y_train,
                            epochs=10,
                            batch_size=512,
                            validation_data=(self.x_val, self.y_val),
                            verbose=1)
        return history

    def test(self):
        results = self.model.evaluate(self.test_examples, self.test_labels)

        print(results)

    def save(self, fp):
        self.model.save(fp)

    def load_model(self, fp):
        self.model = keras.models.load_model(filepath=fp)
        return self.model

if __name__ == "__main__":
    mModel = TextClassifierModel()
    mModel.build()
    mModel.load_IMDB_tfds()
    mModel.train()
    mModel.test()
    mModel.save(fp="model/SavedModel/TextClassificationModel")
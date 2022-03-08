import os
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from MLEXray.EdgeMLMonitor import EdgeMLMonitor
from MLEXray.Model_Runner.Image_Model_Runner import Image_Model_Runner
from MLEXray.Model_Runner.Model_Runner import Model_Runner
from MLEXray.Utils.params import ModelName, DatasetName
from MLEXray.Utils.tf_utils import KerasModel_preprocessing
import cv2
import tensorflow_hub as hub

class Keras_Image_Model_Runner(Image_Model_Runner):
    embed_layer_name = None
    def __init__(self, model_name, eval=False):
        if model_name == ModelName.MobileNetV1:
            self.core = tf.keras.applications.MobileNet(
                input_shape=(224,224,3),
                include_top=True,
                weights='imagenet',
            )
        elif model_name == ModelName.MobileNetV2:
            self.core = tf.keras.applications.MobileNetV2(
                input_shape=(224,224,3),
                include_top=True,
                weights='imagenet',
            )
            # self.embed_layer_name = "global_average_pooling2d"
        elif model_name == ModelName.MobileNetV3_Large:
            self.core = tf.keras.applications.MobileNetV3Large(
                input_shape=(224, 224, 3),
                include_top=True,
                weights='imagenet',
                include_preprocessing=False,
            )
        elif model_name == ModelName.MobileNetV3_Small:
            self.core = tf.keras.applications.MobileNetV3Small(
                input_shape=(224, 224, 3),
                include_top=True,
                weights='imagenet',
                include_preprocessing=False,
            )
        elif model_name == ModelName.InceptionV3:
            self.core = tf.keras.applications.InceptionV3(
                include_top=True,
                weights='imagenet',
            )
        elif model_name == ModelName.DenseNet121:
            self.core = tf.keras.applications.DenseNet121(
                input_shape=(224, 224, 3),
                include_top=True,
                weights='imagenet',
            )
        elif model_name == ModelName.EfficientNetB0:
            self.core = tf.keras.applications.EfficientNetB0(
                include_top=True,
                weights='imagenet',
            )
        elif model_name == ModelName.NASNetMobile:
            self.core = tf.keras.applications.NASNetMobile(
                include_top=True,
                weights='imagenet',
            )
        elif model_name == ModelName.ResNet50V1:
            self.core = tf.keras.applications.ResNet50(
                include_top=True,
                weights='imagenet',
            )
        elif model_name == ModelName.ResNet50V2:
            self.core = tf.keras.applications.ResNet50V2(
                include_top=True,
                weights='imagenet',
            )
        else:
            raise ValueError(f"Model name {self.model_name} not supported yet!")

        # i = tf.keras.layers.Input([None, None, 3], dtype=tf.uint8)
        # x = tf.cast(i, tf.float32)
        # x = tf.keras.applications.mobilenet.preprocess_input(x)
        # x = self.core(x)
        # self.model = tf.keras.Model(inputs=[i], outputs=[x])

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


    def evaluate_on_dataset(self, dataset, model_name, batch_size=128):

        dataset = dataset.map(lambda x: self.preprocess_tfdataset_image(x, model_name, input_size=self.input_size))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        result = self.model.evaluate(dataset)
        print(result)

    def train_on_dataset(self, dataset, batch_size=128, epochs=10):

        dataset=dataset.map(lambda x: self.preprocess_tfdataset_image(x, model_name, input_size=self.input_size))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        result = self.model.fit(dataset, epochs=epochs)
        print(result)

    def preprocess_tfdataset_image(self, data, model_name, input_size=224):
        image = data['image']
        print(image)
        image = tf.cast(image, tf.float32)
        image = KerasModel_preprocessing(image, model_name=model_name)
        image = tf.image.resize(image, [input_size,input_size], method=tf.image.ResizeMethod.AREA)
        print(image)
        label = tf.one_hot(data['label'], depth=1000)

        data['image'] = image
        data['label'] = label
        return data['image'], data['label']


if __name__ == '__main__':
    # run data trace
    # trace_name = "Sat_Aug_07_16-19-05_PDT_2021_Classification"
    # trace_name = "imagenet2012"
    # trace_name = "imagenet2012_100"
    # trace_name = "imagenet2012_10"
    trace_name = DatasetName.Imagenet_1
    data_path = f"data/0_data/{trace_name}/nativeInput/"

    model_name_list = [
                       ModelName.MobileNetV1,
                       ModelName.MobileNetV2,
                       ModelName.MobileNetV3_Large,
                       ModelName.MobileNetV3_Small,
                       ModelName.InceptionV3,
                       ModelName.DenseNet121,
                       # ModelName.EfficientNetB0,
                       # ModelName.ResNet50V1,
                       ModelName.ResNet50V2,
                       # ModelName.NASNetMobile,
                        ]
    # model_name = ModelName.DenseNet121
    # model_name = ModelName.InceptionV3
    # model_name = ModelName.MobileNetV1
    # model_name = ModelName.MobileNetV2
    # model_name = ModelName.MobileNetV3_Large
    # model_name = ModelName.MobileNetV3_Small
    # model_name = ModelName.NASNetMobile
    # model_name = ModelName.ResNet50V1
    # model_name = ModelName.ResNet50V2
    for model_name in model_name_list:
        runner = Keras_Image_Model_Runner(model_name=model_name)
        runner.set_per_layer_logging(True)
        log_path = f"data/trace_{trace_name}/{model_name}/1_cloud/{trace_name}/"

        runner.run_image_data_folder(data_path, log_path)

    # # evaluate on public dataset
    # runner = Model_Runner('MobileNetV2', eval=True)
    # # dataset = tfds.load('imagenet_v2', split='test', shuffle_files=True)
    # dataset = tfds.load('imagenet2012', split='validation', shuffle_files=True)
    # # dataset = tfds.load('mnist', split='test', shuffle_files=True)
    # runner.evaluate_on_dataset(dataset, batch_size=128)
    # # runner.train_on_dataset(dataset)

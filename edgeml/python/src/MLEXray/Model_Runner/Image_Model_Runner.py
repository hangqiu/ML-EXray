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

class Image_Model_Runner(Model_Runner):

    def run_image_data_folder(self, data_path, logDir, enableLog=True):
        """

        :param data_path:
        :param logDir:
        :param enableLog: default as True, set False to skip logging for performance eval
        :return:
        """
        print(f"Running model on dataset: {data_path}")

        if enableLog: self.mMonitor.set_logDir(logDir)
        images = os.listdir(data_path)
        images = sorted(images)
        idx = 0
        # while idx < len(images):
        result = []
        imgmetas = []
        for i in tqdm(range(len(images))):
            image = images[i]
            res, imgmeta = self.run_one_image(data_path + image, enableLog)
            idx += 1
            result.append(res)
            imgmetas.append(imgmeta)
        return images, imgmetas, result

    def run_one_image(self, image_path, enableLog=False):
        if ".jpg" in image_path.lower() or ".jpeg" in image_path.lower():
            raw_tf_image = tf.image.decode_jpeg(tf.io.read_file(image_path))
        elif ".png" in image_path.lower():
            raw_tf_image = tf.image.decode_png(tf.io.read_file(image_path))
        else:
            raise ValueError(f"{image_path} type not supported")
        # print('decoded png')
        # print(image)
        raw_tf_image = np.array(raw_tf_image)
        if raw_tf_image.shape[-1] == 1:
            raw_tf_image = cv2.cvtColor(raw_tf_image, cv2.COLOR_GRAY2RGB)
        # add channel permutation bug
        # raw_tf_image = cv2.cvtColor(raw_tf_image, cv2.COLOR_BGR2RGB)

        raw_image = tf.expand_dims(raw_tf_image, axis=0)
        image = tf.cast(raw_image, tf.float32)
        # norm
        # image = image/2

        image = tf.image.resize(image, [self.input_size, self.input_size], method=tf.image.ResizeMethod.AREA)
        # image = tf.image.resize(image, [self.input_size, self.input_size], method=tf.image.ResizeMethod.BILINEAR)
        image = KerasModel_preprocessing(image, model_name=self.model_name)
        # print(image)
        if enableLog: self.mMonitor.onInferenceStart()
        result = self.invoke_model(image)
        if enableLog: self.mMonitor.onInferenceStop(interpreter=self.interpreter, result=result, raw_input=raw_image)
        img_meta = dict()
        img_meta["size"] = raw_tf_image.shape
        return result, img_meta

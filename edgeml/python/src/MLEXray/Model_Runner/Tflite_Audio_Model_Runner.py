import csv
import zipfile

import tensorflow as tf
import numpy as np
import cv2
import os
from MLEXray.EdgeMLMonitor import EdgeMLMonitor
from MLEXray.Model_Runner.Audio_Model_Runner import Audio_Model_Runner
from MLEXray.Model_Runner.Image_Model_Runner import Image_Model_Runner
from MLEXray.Model_Runner.Model_Runner import Model_Runner
from MLEXray.Utils.params import ModelName, DatasetName

from MLEXray.Utils.tf_utils import KerasModel_preprocessing


class Tflite_Audio_Model_Runner(Audio_Model_Runner):
    interpreter = None
    input_details = None
    output_details = None
    input_size = None
    quantized = None

    def __init__(self,
                 model_name,
                 quantized,
                 tflite_model_dir=None,
                 tflite_model=None,
                 per_layer_logging=False,
                 referenceOp=False
                 ):
        if tflite_model is None and tflite_model_dir is None:
            raise ValueError(f"Either model content or model path has to be provided")
        OpResolver = tf.lite.experimental.OpResolverType.AUTO
        if referenceOp:
            OpResolver = tf.lite.experimental.OpResolverType.BUILTIN_REF
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model,
                                               model_path=tflite_model_dir,
                                               experimental_preserve_all_tensors=True,
                                               experimental_op_resolver_type=OpResolver)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        print("Input Details", self.input_details)
        print("Output Details", self.output_details)
        self.input_size = self.input_details['shape']
        self.quantized = quantized
        print("tflite model built")

        super().__init__(model_name)
        if per_layer_logging:
            self.mMonitor.set_per_layer_output_logging(True)
        else:
            self.mMonitor.set_per_layer_output_logging(False)

    def set_per_layer_logging(self, value):
        self.mMonitor.set_per_layer_output_logging(value)

    def invoke_model(self, input):
        if self.quantized:
            # Manually quantize the input from float to integer
            scale, zero_point = self.input_details['quantization']
            print(f"Input details: scale:{scale}, zero_point:{zero_point}")
            tflite_integer_input = input / scale + zero_point
            tflite_integer_input = tf.cast(tflite_integer_input, self.input_details['dtype'])
            # print(tflite_integer_input.shape)
            tf_input = tflite_integer_input
        else:
            tf_input = input

        # yamnet default input size is [1], need to resize it according to wave length
        self.interpreter.resize_tensor_input(self.input_details['index'], [input.shape[0]], strict=True)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details['index'], tf_input)
        self.interpreter.invoke()
        tflite_output = self.interpreter.get_tensor(self.output_details['index'])
        if self.quantized:
            # Manually dequantize the output from integer to float
            scale, zero_point = self.output_details['quantization']
            print(f"Output details: scale:{scale}, zero_point:{zero_point}")
            tf_output = tflite_output.astype(np.float32)
            tf_output = (tf_output - zero_point) * scale
        else:
            tf_output = tflite_output

        result = dict()
        result['input'] = tf_input.numpy()
        result['predictions'] = tf_output

        tensor_details = self.interpreter.get_tensor_details()
        for tensor in tensor_details:
            if tensor['name'] != "":
                try:
                    value = self.interpreter.get_tensor(tensor['index'])
                    result[tensor['name']] = value
                except ValueError as e:
                    pass
        # print(result.keys())
        return result

def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

def test_audio_models():
    trace_name = DatasetName.AudioSet
    model_name = ModelName.YamNet
    tflite_path = "model/TfliteModels_TFHub/yamnet_audioset_16000.tflite"

    referenceOp = False
    per_layer_logging = False

    log_path = f"data/trace_{trace_name}/{model_name}/1_cloud_tflite/{trace_name}/"
    if referenceOp:
        log_path = f"data/trace_{trace_name}/{model_name}/1_cloud_tflite_refOp/{trace_name}/"
    runner = Tflite_Audio_Model_Runner(model_name=model_name,
                                       quantized=False,
                                       tflite_model_dir=tflite_path,
                                       per_layer_logging=per_layer_logging,
                                       referenceOp=referenceOp)
    # audio_path = "data/0_data/mini_speech_commands/down/0a9f9af7_nohash_0.wav"
    audio_path = "data/0_data/speech_whistling2.wav"
    # audio_path = "data/0_data/miaow_16k.wav"
    res = runner.run_one_audio(audio_path)
    pred = res['predictions']
    pred = pred.mean(axis=0)
    print(pred.shape)

    top_class_index = pred.argmax()
    # f = open('model/yamnet_class_map.csv')
    # csv_text = f.read()
    class_names = class_names_from_csv('model/yamnet_class_map.csv')
    print(len(class_names))
    # labels_file = zipfile.ZipFile(tflite_path).open('yamnet_label_list.txt')
    # labels = [l.decode('utf-8').strip() for l in labels_file.readlines()]

    print(class_names[top_class_index])  # Should print 'Silence'.

if __name__ == '__main__':
    test_audio_models()

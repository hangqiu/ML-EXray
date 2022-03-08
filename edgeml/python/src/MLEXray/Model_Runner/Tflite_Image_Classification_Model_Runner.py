import tensorflow as tf
import numpy as np
import cv2
import os
from MLEXray.EdgeMLMonitor import EdgeMLMonitor
from MLEXray.Model_Runner.Image_Model_Runner import Image_Model_Runner
from MLEXray.Model_Runner.Model_Runner import Model_Runner
from MLEXray.Utils.params import ModelName, DatasetName

from MLEXray.Utils.tf_utils import KerasModel_preprocessing


class Tflite_Image_Classification_Model_Runner(Image_Model_Runner):
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
        self.input_details = self.interpreter.get_input_details()[0] # assume only 1 input, image
        self.output_details = self.interpreter.get_output_details()[0] # assume only 1 output, predictions
        # print("Input Details", self.input_details)
        # print("Output Details", self.output_details)
        self.input_size = self.input_details['shape'][-2]
        self.quantized = quantized
        # print("tflite model built")

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
            # print(f"Input details: scale:{scale}, zero_point:{zero_point}")
            tflite_integer_input = input / scale + zero_point
            tflite_integer_input = tf.cast(tflite_integer_input, self.input_details['dtype'])
            # print(tflite_integer_input.shape)
            tf_input = tflite_integer_input
        else:
            tf_input = input

        self.interpreter.set_tensor(self.input_details['index'], tf_input)
        self.interpreter.invoke()
        tflite_output = self.interpreter.get_tensor(self.output_details['index'])
        if self.quantized:
            # Manually dequantize the output from integer to float
            scale, zero_point = self.output_details['quantization']
            # print(f"Output details: scale:{scale}, zero_point:{zero_point}")
            tf_output = tflite_output.astype(np.float32)
            tf_output = (tf_output - zero_point) * scale
        else:
            tf_output = tflite_output

        result = dict()
        result['input'] = tf_input.numpy()
        result['predictions'] = tf_output

        # tensor_details = self.interpreter.get_tensor_details()
        # for tensor in tensor_details:
        #     if tensor['name'] != "":
        #         value = self.interpreter.get_tensor(tensor['index'])
        #         # result[f"{tensor['index']}_{tensor['name']}"] = value
        #         result[tensor['name']] = value
        # print(result.keys())
        return result



def test_image_models():
    # run data trace
    # trace_name = "imagenet2012"
    # trace_name = "imagenet2012_10"
    trace_name = DatasetName.Imagenet_1
    # trace_name = DatasetName.Imagenet_100
    data_path = f"data/0_data/{trace_name}/nativeInput/"

    per_layer_logging = True
    # per_layer_logging = False
    referenceOp = True
    # referenceOp = False

    model_name_list = [
        # ModelName.MobileNetV1,
        ModelName.MobileNetV2,
        # ModelName.MobileNetV3_Large,
        # ModelName.MobileNetV3_Small,
        # ModelName.InceptionV3,
        # ModelName.DenseNet121,
        # ModelName.EfficientNetB0,
        # ModelName.ResNet50V1,
        # ModelName.ResNet50V2,
        # ModelName.NASNetMobile,
    ]

    for model_name in model_name_list:
        tflite_path = f"model/ConvertedModels/{model_name}/{model_name}_imagenet_224.tflite"
        if model_name == ModelName.InceptionV3:
            tflite_path = f"model/ConvertedModels/{model_name}/{model_name}_imagenet_299.tflite"
        log_path = f"data/trace_{trace_name}/{model_name}/1_cloud_tflite/{trace_name}/"
        if referenceOp:
            log_path = f"data/trace_{trace_name}/{model_name}/1_cloud_tflite_refOp/{trace_name}/"
        runner = Tflite_Image_Classification_Model_Runner(model_name=model_name,
                                                          quantized=False,
                                                          tflite_model_dir=tflite_path,
                                                          per_layer_logging=per_layer_logging,
                                                          referenceOp=referenceOp)
        runner.run_image_data_folder(data_path, log_path, enableLog=True)

        tflite_path = f"model/ConvertedModels/{model_name}/{model_name}_imagenet_quant_224.tflite"
        if model_name == ModelName.InceptionV3:
            tflite_path = f"model/ConvertedModels/{model_name}/{model_name}_imagenet_quant_299.tflite"
        log_path = f"data/trace_{trace_name}/{model_name}/1_cloud_tflite_quant/{trace_name}/"
        if referenceOp:
            log_path = f"data/trace_{trace_name}/{model_name}/1_cloud_tflite_quant_refOp/{trace_name}/"
        runner = Tflite_Image_Classification_Model_Runner(model_name=model_name,
                                                          quantized=True,
                                                          tflite_model_dir=tflite_path,
                                                          per_layer_logging=per_layer_logging,
                                                          referenceOp=referenceOp)
        runner.run_image_data_folder(data_path, log_path, enableLog=True)


if __name__ == '__main__':
    test_image_models()

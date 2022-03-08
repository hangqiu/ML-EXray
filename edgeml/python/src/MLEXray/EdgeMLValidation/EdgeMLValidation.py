import os
import tensorflow as tf

from MLEXray.ML_Diff.TopKAccuracy import TopKAccuracy
from MLEXray.Model_Runner.Tflite_Image_Classification_Model_Runner import Tflite_Image_Classification_Model_Runner
from MLEXray.ML_Diff.MLLogReader import MLLogReader
from MLEXray.ML_Diff.MLDiffer import MLDiffer
from MLEXray.Utils.params import ScalarLogKeys, VectorLogKeys, ModelName


class EdgeMLValidation:
    """
    The basic user interface for mobile ML deployment validation.
    """

    default_per_layer_validation_data = "data/0_data/imagenet2012_1/nativeInput/"
    default_per_layer_tmp_dir = "data/validation/imagenet2012_1/"

    default_validation_data = "data/0_data/imagenet2012_100/nativeInput/"
    default_tmp_dir = "data/validation/imagenet2012_100/"
    default_validation_setsize = 100
    # default_validation_data = "data/0_data/imagenet2012_10/nativeInput/"
    # default_tmp_dir = "data/validation/imagenet2012_10/"
    # default_validation_setsize = 10
    default_groundtruth_filepath = "data/0_data/imagenet2012/groundtruth_label.txt"
    default_model_output_label = "model/imagenet_label_1000.txt"

    mReader=None
    mDiffer=None
    def __init__(self,
                 mobile_log_path,
                 mobile_per_layer_log_path,
                 tflite_path,
                 model_name,
                 quantized=False,
                 per_layer_logging=False,
                 referenceOp=False):
        """

        :param mobile_log_path: assume the same validation dataset is run on mobile and log is downloaded to this local path
        :param tflite_path: the model used on mobile test
        :param model_name:
        :param quantized: True, if it's a quantized model
        :param per_layer_logging: True, if enable per layer debugging
        """
        if not os.path.exists(self.default_tmp_dir):
            os.makedirs(self.default_tmp_dir)
        if not os.path.exists(self.default_per_layer_tmp_dir):
            os.makedirs(self.default_per_layer_tmp_dir)

        self._mobile_log_path = mobile_log_path
        self._mobile_per_layer_log_path = mobile_per_layer_log_path
        # replace with the data run on mobile side, if exist
        # mobile_log_input_data = self._mobile_log_path + "/nativeInput/"
        # if os.path.exists(mobile_log_input_data):
        #     self.default_validation_data = mobile_log_input_data
        self._tflite_path = tflite_path
        self._model_name = model_name
        self._validation_log_path = self.default_tmp_dir + self._model_name
        self._validation_per_layer_log_path = self.default_per_layer_tmp_dir + self._model_name
        self._per_layer_logging = per_layer_logging
        self._tflite_model_runner = Tflite_Image_Classification_Model_Runner(
            model_name=model_name,
            quantized=quantized,
            tflite_model_dir=self._tflite_path,
            per_layer_logging=per_layer_logging,
            referenceOp=referenceOp # by default, enable reference op for debugging
        )
        self.mDiffer=MLDiffer()

    def check_accuracy(self, scalar_key_list, vector_key_list):
        # run the validation data in python interpreter
        print("Running Tflite model for accuracy evaluation")
        self._tflite_model_runner.set_per_layer_logging(False)
        self._tflite_model_runner.run_image_data_folder(
            data_path=self.default_validation_data,
            logDir=self._validation_log_path
        )
        print("Reading validation set logs")
        mobile_log = self.mReader.read_log(self._mobile_log_path)
        print(mobile_log.keys())
        validation_log = self.mReader.read_log(self._validation_log_path)
        print(validation_log.keys())

        # First, eval end-to-end accuracy
        print("Accuracy Evaluation")
        mTopK = TopKAccuracy(
            groundtruth_filepath=self.default_groundtruth_filepath,
            model_output_label=self.default_model_output_label,
            num_samples=self.default_validation_setsize
        )
        mobile_inf_res = mobile_log[VectorLogKeys.TopKResults]
        mTopK.parse_inference_result(mobile_inf_res)
        accuracy = mTopK.eval_accuracy(topk=3)
        print("Mobile Accuracy:", accuracy)

        validation_inf_res = validation_log[VectorLogKeys.TopKResults]
        mTopK.parse_inference_result(validation_inf_res)
        accuracy = mTopK.eval_accuracy(topk=3)
        print("Validation Reference Accuracy:", accuracy)

        # Next, eval MLDiff
        print("MLDiff")
        self.mDiffer.diff(
            trace_list=[mobile_log, validation_log],
            name_list=['Mobile', 'Validation'],
            scalar_key_list=scalar_key_list,
            vector_key_list=vector_key_list,
            output_dir=self.default_tmp_dir
        )

    def per_layer_debugging(self, per_layer_vector_key_list):
        # run the validation data in python interpreter
        print("Running Tflite model for per-layer debugging")
        self._tflite_model_runner.set_per_layer_logging(True)
        self._tflite_model_runner.run_image_data_folder(
            data_path=self.default_per_layer_validation_data,
            logDir=self._validation_per_layer_log_path
        )
        print("Reading per-layer logs")
        mobile_per_layer_log = self.mReader.read_log(self._mobile_per_layer_log_path)
        validation_per_layer_log = self.mReader.read_log(self._validation_per_layer_log_path)

        print("MLDiff")
        self.mDiffer.diff(
            trace_list=[mobile_per_layer_log, validation_per_layer_log],
            name_list=['Mobile', 'Validation'],
            scalar_key_list=None,
            vector_key_list=per_layer_vector_key_list,
            tflite_model_list=[self._tflite_path, self._tflite_path],
            output_dir=self.default_per_layer_tmp_dir
        )

    def run(self):
        scalar_key_list = [ScalarLogKeys.InferenceLatency]
        vector_key_list = [VectorLogKeys.ModelInput,
                           VectorLogKeys.ModelOutput,
                           VectorLogKeys.TopKResults]
        per_layer_vector_key_list=vector_key_list
        # add model tensors to log keys
        interpreter = tf.lite.Interpreter(model_path=self._tflite_path)
        tensor_details = interpreter.get_tensor_details()
        for tensor in tensor_details:
            tensor_name = tensor['name']
            per_layer_vector_key_list.append(tensor_name)

        self.mReader = MLLogReader(
            addtional_scalar_list=scalar_key_list,
            addtional_vector_list=per_layer_vector_key_list)

        self.check_accuracy(scalar_key_list, vector_key_list)
        self.per_layer_debugging(per_layer_vector_key_list)


        # TODO: report numerical differences and location

        # TODO: apply user defined assertions

if __name__ == "__main__":
    mobile_log_path="data/trace_imagenet2012_100/MobileNetV2/2_tflite/imagenet2012_100_playback_at_Mon_Aug_09_01-18-46_PDT_2021_classification"
    # mobile_log_path="data/trace_imagenet2012_10/MobileNetV2/2_tflite/imagenet2012_10_playback_at_Thu_Sep_16_10-57-54_PDT_2021_classification"
    mobile_per_layer_log_path = "data/trace_imagenet2012_1/MobileNetV2/2_tflite/imagenet2012_1_playback_at_Thu_Sep_16_11-45-29_PDT_2021_classification"
    tflite_path="model/ConvertedModels/MobileNetV2/MobileNetV2_imagenet_224.tflite"
    model_name=ModelName.MobileNetV2
    quantized = False
    per_layer_logging = False
    referenceOp = False
    mValidation = EdgeMLValidation(
        mobile_log_path=mobile_log_path,
        mobile_per_layer_log_path=mobile_per_layer_log_path,
        tflite_path=tflite_path,
        model_name=model_name,
        quantized=quantized,
        per_layer_logging=per_layer_logging,
        referenceOp=referenceOp
    )

    mValidation.run()
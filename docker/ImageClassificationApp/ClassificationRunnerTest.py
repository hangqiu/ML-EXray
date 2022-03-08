import time
from MLEXray.Utils.params import ModelName, VectorLogKeys, DatasetName
from MLEXray.Model_Runner.Tflite_Image_Classification_Model_Runner import Tflite_Image_Classification_Model_Runner


# run data trace

trace_name = DatasetName.Imagenet_1
data_path = "./data/"
trace_path = "./trace/"
model_path = "./model/"


per_layer_logging = True
referenceOp = True

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

while True:
    print("Running image classification")
    for model_name in model_name_list:
        tflite_path = f"{model_path}/{model_name}/{model_name}_imagenet_224.tflite"
        if model_name == ModelName.InceptionV3:
            tflite_path = f"{model_path}/{model_name}/{model_name}_imagenet_299.tflite"
        log_path = f"{trace_path}/trace_{trace_name}/{model_name}/1_cloud_tflite/{trace_name}/"
        if referenceOp:
            log_path = f"{trace_path}/trace_{trace_name}/{model_name}/1_cloud_tflite_refOp/{trace_name}/"
        runner = Tflite_Image_Classification_Model_Runner(model_name=model_name,
                                                          quantized=False,
                                                          tflite_model_dir=tflite_path,
                                                          per_layer_logging=per_layer_logging,
                                                          referenceOp=referenceOp)
        runner.run_image_data_folder(data_path, log_path, enableLog=True)

        tflite_path = f"{model_path}/{model_name}/{model_name}_imagenet_quant_224.tflite"
        if model_name == ModelName.InceptionV3:
            tflite_path = f"{model_path}/{model_name}/{model_name}_imagenet_quant_299.tflite"
        log_path = f"{trace_path}/trace_{trace_name}/{model_name}/1_cloud_tflite_quant/{trace_name}/"
        if referenceOp:
            log_path = f"{trace_path}/trace_{trace_name}/{model_name}/1_cloud_tflite_quant_refOp/{trace_name}/"
        runner = Tflite_Image_Classification_Model_Runner(model_name=model_name,
                                                          quantized=True,
                                                          tflite_model_dir=tflite_path,
                                                          per_layer_logging=per_layer_logging,
                                                          referenceOp=referenceOp)
        runner.run_image_data_folder(data_path, log_path, enableLog=True)

    time.sleep(10)
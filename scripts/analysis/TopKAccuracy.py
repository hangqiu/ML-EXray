from edgeml.python.src.MLEXray.ML_Diff.MLLogReader import MLLogReader
from edgeml.python.src.MLEXray.ML_Diff.TopKAccuracy import get_log_path, TopKAccuracy
from edgeml.python.src.MLEXray.Utils.params import ModelName, ScalarLogKeys, VectorLogKeys

groundtruth_filepath = "data/0_data/imagenet2012/groundtruth_label.txt"
model_output_label = "model/imagenet_label_1000.txt"


if __name__=="__main__":

    trace_name = "imagenet2012_100"

    # model_name = ModelName.MobileNetV1
    model_name = ModelName.MobileNetV2
    # model_name = ModelName.MobileNetV3_Large
    # model_name = ModelName.MobileNetV3_Small
    # model_name = ModelName.NASNetMobile
    # model_name = ModelName.InceptionV3
    # model_name = ModelName.DenseNet121
    # model_name = ModelName.ResNet50V1
    # model_name = ModelName.ResNet50V2

    log_paths = get_log_path(f"data/trace_{trace_name}/{model_name}")

    scalar_key_list = [ScalarLogKeys.InferenceLatency]
    vector_key_list = [VectorLogKeys.ModelInput,
                       VectorLogKeys.ModelOutput,
                       VectorLogKeys.TopKResults]

    mReader = MLLogReader(addtional_scalar_list=scalar_key_list,
                          addtional_vector_list=vector_key_list)
    mTopK = TopKAccuracy(
        groundtruth_filepath=groundtruth_filepath,
        model_output_label=model_output_label
    )
    for log_path in log_paths:
        print(log_path)
        pd_data = mReader.read_log(log_path)
        print(pd_data)
        inf_res = pd_data['Inference Result']
        mTopK.parse_inference_result(inf_res)
        accuracy = mTopK.eval_accuracy(topk=3)
        print(accuracy)
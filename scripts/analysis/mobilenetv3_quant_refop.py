import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from edgeml.python.src.MLEXray.ML_Diff.MLDiffer import MLDiffer
from edgeml.python.src.MLEXray.ML_Diff.MLLogReader import MLLogReader
from edgeml.python.src.MLEXray.Utils.params import ModelName, DatasetName
from edgeml.python.src.MLEXray.plots.per_layer_plots import get_norm_rmse_from_csv, plot_one_model


model_name = ModelName.MobileNetV3_Small
quant_log_dir = f"./analysis/{model_name}/quant_vs_cloudQuant/"
quant_refOp_log_dir = f"./analysis/{model_name}/quantRef_vs_cloudQuant/"


def parse_and_compare():

    trace_name = DatasetName.Imagenet_1
    tflite_quant_model_dir = ModelName.get_tflite_model_path(model_name, quantized=True)

    tflite_cloud_trace = f"data/trace_{trace_name}/{model_name}/1_cloud_tflite_refOp/{trace_name}/"
    tflite_quant_refop_cloud_trace = f"data/trace_{trace_name}/{model_name}/1_cloud_tflite_quant_refOp/{trace_name}/"
    tflite_quant_trace = "data/trace_imagenet2012_1/MobileNetV3_Small/3_tflite_quant/imagenet2012_1_playback_at_Mon_Oct_11_16-06-20_PDT_2021_classification"
    tflite_quant_refop_trace = "data/trace_imagenet2012_1/MobileNetV3_Small/3_tflite_quant_refOp/imagenet2012_1_playback_at_Mon_Oct_11_16-35-24_PDT_2021_classification"

    interpreter = tf.lite.Interpreter(model_path=tflite_quant_model_dir)
    tensor_details = interpreter.get_tensor_details()
    vector_key_list = []
    for tensor in tensor_details:
        tensor_name = tensor['name']
        vector_key_list.append(tensor_name)
    mReader = MLLogReader(addtional_vector_list=vector_key_list)

    pd_tflite_cloud = mReader.read_log(tflite_cloud_trace)
    pd_tflite_quant = mReader.read_log(tflite_quant_trace)
    pd_tflite_quant_cloud = mReader.read_log(tflite_quant_refop_cloud_trace)
    pd_tflite_quant_refop = mReader.read_log(tflite_quant_refop_trace)

    print(pd_tflite_cloud.keys())
    print(pd_tflite_quant.keys())
    print(pd_tflite_quant_cloud.keys())
    print(pd_tflite_quant_refop.keys())


    # quant vs cloud quant

    name_list = [f"{model_name}_TFLite_Quant", f"{model_name}_TFLite_Quant_Cloud"]
    trace_list = [pd_tflite_quant, pd_tflite_quant_cloud]
    tflite_model_list = [tflite_quant_model_dir, tflite_quant_model_dir]
    mDiffer = MLDiffer()
    mDiffer.diff(trace_list, name_list, scalar_key_list=None,
                 vector_key_list=vector_key_list, tflite_model_list=tflite_model_list, output_dir=quant_log_dir)

    # quant ref Op vs cloud quant
    name_list = [f"{model_name}_TFLite_Quant_RefOp", f"{model_name}_TFLite_Quant_Cloud"]
    trace_list = [pd_tflite_quant_refop, pd_tflite_quant_cloud]
    tflite_model_list = [tflite_quant_model_dir, tflite_quant_model_dir]
    mDiffer.diff(trace_list, name_list, scalar_key_list=None,
                 vector_key_list=vector_key_list, tflite_model_list=tflite_model_list, output_dir=quant_refOp_log_dir)



def plot_results():

    mobilenetv3_quant_vs_CloudQuantRef = f"{quant_log_dir}/0_0_debug_report.csv"
    mobilenetv3_quantRef_vs_CloudQuantRef = f"{quant_refOp_log_dir}/0_0_debug_report.csv"

    mobilenetv3_start = 114
    mobilenetv3_quant_vs_CloudQuantRef_pd = get_norm_rmse_from_csv(fp=mobilenetv3_quant_vs_CloudQuantRef,
                                                                   output_row_start=mobilenetv3_start,
                                                                   # output_row_end=length + mobilenetv3_start
                                                                   )
    mobilenetv3_quantRef_vs_CloudQuantRef_pd = get_norm_rmse_from_csv(fp=mobilenetv3_quantRef_vs_CloudQuantRef,
                                                                      output_row_start=mobilenetv3_start,
                                                                      # output_row_end=length + mobilenetv3_start
                                                                      )

    quant_vs_CloudQuantRef_fig = "quant_vs_CloudQuantRef_rmse_mobilenetv3_small.png"
    plot_one_model(mobilenetv3_quant_vs_CloudQuantRef_pd, mobilenetv3_quantRef_vs_CloudQuantRef_pd,
                   ModelName.MobileNetV3_Small, quant_vs_CloudQuantRef_fig)

if __name__=="__main__":
    parse_and_compare()
    plot_results()

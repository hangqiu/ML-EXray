import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from edgeml.python.src.MLEXray.ML_Diff.MLDiffer import MLDiffer
from edgeml.python.src.MLEXray.ML_Diff.MLLogReader import MLLogReader
from edgeml.python.src.MLEXray.Utils.params import ModelName, DatasetName
from edgeml.python.src.MLEXray.plots.per_layer_plots import get_norm_rmse_from_csv, plot_one_model


def parse_and_compare():
    # model_name = ModelName.MobileNetV2
    # model_name = ModelName.DenseNet121
    # model_name = ModelName.ResNet50V2
    model_name = ModelName.MobileNetV3_Small
    # model_name = ModelName.MobileNetV1

    trace_name = DatasetName.Imagenet_1
    data_trace = f"data/0_data/{trace_name}"

    tflite_unquant_model_dir = ModelName.get_tflite_model_path(model_name, quantized=False)
    tflite_quant_model_dir = ModelName.get_tflite_model_path(model_name, quantized=True)

    tflite_cloud_trace = f"data/trace_{trace_name}/{model_name}/1_cloud_tflite_refOp/{trace_name}/"
    tflite_quant_refop_cloud_trace = f"data/trace_{trace_name}/{model_name}/1_cloud_tflite_quant_refOp/{trace_name}/"

    # # MobileNet V2
    # tflite_trace = "data/trace_imagenet2012_1/MobileNetV2/2_tflite/imagenet2012_1_playback_at_Wed_Oct_06_23-46-15_PDT_2021_classification"
    # tflite_quant_trace = "data/trace_imagenet2012_1/MobileNetV2/3_tflite_quant/imagenet2012_1_playback_at_Wed_Oct_06_23-47-55_PDT_2021_classification"
    # tflite_quant_refop_trace = "data/trace_imagenet2012_1/MobileNetV2/3_tflite_quant_refOp/imagenet2012_1_playback_at_Mon_Oct_11_14-03-47_PDT_2021_classification"

    # MobileNet V3 small
    tflite_quant_trace = "data/trace_imagenet2012_1/MobileNetV3_Small/3_tflite_quant/imagenet2012_1_playback_at_Mon_Oct_11_16-06-20_PDT_2021_classification"
    tflite_quant_refop_trace = "data/trace_imagenet2012_1/MobileNetV3_Small/3_tflite_quant_refOp/imagenet2012_1_playback_at_Mon_Oct_11_16-35-24_PDT_2021_classification"

    # DenseNet121
    # tflite_trace = "data/trace_imagenet2012_1/DenseNet121/2_tflite/imagenet2012_1_playback_at_Wed_Oct_06_23-54-54_PDT_2021_classification"
    # tflite_quant_trace = "data/trace_imagenet2012_1/DenseNet121/3_tflite_quant/imagenet2012_1_playback_at_Wed_Oct_06_23-56-58_PDT_2021_classification"
    # tflite_quant_refop_trace = "data/trace_imagenet2012_1/DenseNet121/3_tflite_quant_refOp/imagenet2012_1_playback_at_Mon_Oct_11_14-08-18_PDT_2021_classification"

    # Resnetv2
    # tflite_trace = "data/trace_imagenet2012_1/ResNet50V2/2_tflite/imagenet2012_1_playback_at_Wed_Oct_06_23-48-32_PDT_2021_classification"
    # tflite_quant_trace = "data/trace_imagenet2012_1/ResNet50V2/3_tflite_quant/imagenet2012_1_playback_at_Wed_Oct_06_23-50-26_PDT_2021_classification"
    # tflite_quant_refop_trace = "data/trace_imagenet2012_1/ResNet50V2/3_tflite_quant_refOp/imagenet2012_1_playback_at_Mon_Oct_11_14-04-56_PDT_2021_classification"

    interpreter = tf.lite.Interpreter(model_path=tflite_quant_model_dir)
    tensor_details = interpreter.get_tensor_details()
    # print(tensor_details)
    vector_key_list = []
    for tensor in tensor_details:
        tensor_name = tensor['name']
        vector_key_list.append(tensor_name)

    mReader = MLLogReader(addtional_vector_list=vector_key_list)

    # pd_tflite = mReader.read_log(tflite_trace)
    pd_tflite_cloud = mReader.read_log(tflite_cloud_trace)
    pd_tflite_quant = mReader.read_log(tflite_quant_trace)
    pd_tflite_quant_cloud = mReader.read_log(tflite_quant_refop_cloud_trace)
    pd_tflite_quant_refop = mReader.read_log(tflite_quant_refop_trace)

    # print(pd_tflite.keys())
    print(pd_tflite_cloud.keys())
    print(pd_tflite_quant.keys())
    print(pd_tflite_quant_cloud.keys())
    print(pd_tflite_quant_refop.keys())

    # output_dir = f"./analysis/{model_name}/quant_vs_tflite/"
    # name_list = [f"{model_name}_TFLite_Quant", f"{model_name}_TFLite"]
    # trace_list = [pd_tflite_quant, pd_tflite_cloud]
    # tflite_model_list = [tflite_quant_model_dir, tflite_unquant_model_dir]

    # output_dir = f"./analysis/{model_name}/quantRef_vs_tflite/"
    # name_list = [f"{model_name}_TFLite_Quant_RefOp", f"{model_name}_TFLite"]
    # trace_list = [pd_tflite_quant_refop, pd_tflite_cloud]
    # tflite_model_list = [tflite_quant_model_dir, tflite_unquant_model_dir]

    # output_dir = f"./analysis/{model_name}/tflite_vs_cloudQuant/"
    # name_list = [f"{model_name}_TFLite", f"{model_name}_TFLite_Quant"]
    # trace_list = [pd_tflite, pd_tflite_quant_cloud]
    # tflite_model_list = [tflite_unquant_model_dir, tflite_quant_model_dir]

    output_dir = f"./analysis/{model_name}/quant_vs_cloudQuant/"
    name_list = [f"{model_name}_TFLite_Quant", f"{model_name}_TFLite_Quant_Cloud"]
    trace_list = [pd_tflite_quant, pd_tflite_quant_cloud]
    tflite_model_list = [tflite_quant_model_dir, tflite_quant_model_dir]

    # output_dir = f"./analysis/{model_name}/quantRef_vs_cloudQuant/"
    # name_list = [f"{model_name}_TFLite_Quant_RefOp", f"{model_name}_TFLite_Quant_Cloud"]
    # trace_list = [pd_tflite_quant_refop, pd_tflite_quant_cloud]
    # tflite_model_list = [tflite_quant_model_dir, tflite_quant_model_dir]

    mDiffer = MLDiffer()
    mDiffer.diff(trace_list, name_list, scalar_key_list=None,
                 vector_key_list=vector_key_list, tflite_model_list=tflite_model_list, output_dir=output_dir)

    return output_dir


def plot_results(output_dir):

    # length = 42
    mobilenetv2_quant_vs_CloudQuantRef = "python/plots/analysis_results/MobileNetV2/quant_vs_cloudQuant/0_0_debug_report.csv"
    mobilenetv2_quantRef_vs_CloudQuantRef = "python/plots/analysis_results/MobileNetV2/quantRef_vs_cloudQuant/0_0_debug_report.csv"

    mobilenetv2_quant_vs_CloudQuantRef = "python/plots/analysis_results/MobileNetV2/quant_vs_tflite/0_0_debug_report.csv"
    mobilenetv2_quantRef_vs_CloudQuantRef = "python/plots/analysis_results/MobileNetV2/quantRef_vs_tflite/0_0_debug_report.csv"

    mobilenetv2_start = 109
    mobilenetv2_quant_vs_CloudQuantRef_pd = get_norm_rmse_from_csv(fp=mobilenetv2_quant_vs_CloudQuantRef,
                                                                   output_row_start=mobilenetv2_start,
                                                                   # output_row_end=length + mobilenetv2_start
                                                                   )
    mobilenetv2_quantRef_vs_CloudQuantRef_pd = get_norm_rmse_from_csv(fp=mobilenetv2_quantRef_vs_CloudQuantRef,
                                                                      output_row_start=mobilenetv2_start,
                                                                      # output_row_end=length + mobilenetv2_start
                                                                      )

    mobilenetv3_quant_vs_CloudQuantRef = "python/plots/analysis_results/MobileNetV3_Small/quant_vs_cloudQuant/0_0_debug_report.csv"
    mobilenetv3_quantRef_vs_CloudQuantRef = "python/plots/analysis_results/MobileNetV3_Small/quantRef_vs_cloudQuant/0_0_debug_report.csv"

    mobilenetv3_quant_vs_CloudQuantRef = "python/plots/analysis_results/MobileNetV3_Small/quant_vs_tflite/0_0_debug_report.csv"
    mobilenetv3_quantRef_vs_CloudQuantRef = "python/plots/analysis_results/MobileNetV3_Small/quantRef_vs_tflite/0_0_debug_report.csv"

    mobilenetv3_start = 114
    mobilenetv3_quant_vs_CloudQuantRef_pd = get_norm_rmse_from_csv(fp=mobilenetv3_quant_vs_CloudQuantRef,
                                                                   output_row_start=mobilenetv3_start,
                                                                   # output_row_end=length + mobilenetv3_start
                                                                   )
    mobilenetv3_quantRef_vs_CloudQuantRef_pd = get_norm_rmse_from_csv(fp=mobilenetv3_quantRef_vs_CloudQuantRef,
                                                                      output_row_start=mobilenetv3_start,
                                                                      # output_row_end=length + mobilenetv3_start
                                                                      )

    quant_vs_CloudQuantRef_fig = "python/plots/quant_vs_CloudQuantRef_rmse_mobilenetv2.png"
    plot_one_model(mobilenetv2_quant_vs_CloudQuantRef_pd, mobilenetv2_quantRef_vs_CloudQuantRef_pd,
                   ModelName.MobileNetV2, quant_vs_CloudQuantRef_fig)
    quant_vs_CloudQuantRef_fig = "python/plots/quant_vs_CloudQuantRef_rmse_mobilenetv3_small.png"
    plot_one_model(mobilenetv3_quant_vs_CloudQuantRef_pd, mobilenetv3_quantRef_vs_CloudQuantRef_pd,
                   ModelName.MobileNetV3_Small, quant_vs_CloudQuantRef_fig)

if __name__=="__main__":
    output_dir = parse_and_compare()
    plot_results(output_dir)

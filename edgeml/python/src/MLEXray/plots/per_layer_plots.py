import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from MLEXray.Utils.params import ModelName, PipelineName


def get_norm_rmse_from_csv(fp, output_row_start, output_row_end=-1):
    data = pd.read_csv(fp)
    # print(data)
    data = data[output_row_start:output_row_end]
    # print(data)
    # print(data.keys())

    scale_m = data['norm_max_m'] - data['norm_min_m']
    scale_n = data['norm_max_n'] - data['norm_min_n']
    scale = pd.concat([scale_m, scale_n],axis=1).max(axis=1)
    print(scale)
    data['scale'] = scale

    data['normRMSE'] = data['norm MSE'] ** (1 / 2) / data['scale']
    data = data.reset_index()
    # print(data)
    return data['normRMSE']
    # ret_data = pd.DataFrame({'index': data.index, 'normRMSE': data['normRMSE']})
    # ret_data['index'] = data.index
    # ret_data['normRMSE'] = data['normRMSE']
    # print(ret_data)
    # return ret_data


def plot_refOp_vs_refTflite_rmse_fig():
    mobilenetv3_tflite_vs_CloudQuantRef = "python/plots/analysis_results/MobileNetV3_Small/tflite_vs_cloudQuant/0_0_debug_report.csv"
    output_start = 114
    pd_data = get_norm_rmse_from_csv(fp=mobilenetv3_tflite_vs_CloudQuantRef,
                                     output_row_start=output_start)
    refOp_vs_refTflite_rmse_fig = "python/plots/refOp_vs_refTflite_rmse.png"
    # plot_norm_RMSE(norm_rmse=pd_data, output_dir=refOp_vs_refTflite_rmse_fig)
    plt.figure()
    print(pd_data)
    sns.lineplot(data=pd_data, markers=True)
    plt.xlabel("Layer Index")
    plt.ylabel("Normalized rMSE")
    plt.legend("MobileNetv3 Small")
    plt.title("Mobile Quant Ref vs. Mobile")
    plt.savefig(refOp_vs_refTflite_rmse_fig)


def plot_quant_vs_CloudQuantRef_fig():
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
    plot_one_model(mobilenetv2_quant_vs_CloudQuantRef_pd, mobilenetv2_quantRef_vs_CloudQuantRef_pd, ModelName.MobileNetV2, quant_vs_CloudQuantRef_fig)
    quant_vs_CloudQuantRef_fig = "python/plots/quant_vs_CloudQuantRef_rmse_mobilenetv3_small.png"
    plot_one_model(mobilenetv3_quant_vs_CloudQuantRef_pd, mobilenetv3_quantRef_vs_CloudQuantRef_pd,
                   ModelName.MobileNetV3_Small, quant_vs_CloudQuantRef_fig)


def plot_one_model(mobilenetv2_quant_vs_CloudQuantRef_pd, mobilenetv2_quantRef_vs_CloudQuantRef_pd, model_name, quant_vs_CloudQuantRef_fig):

    quant_vs_CloudQuantRef_data = pd.DataFrame(
        [mobilenetv2_quant_vs_CloudQuantRef_pd, mobilenetv2_quantRef_vs_CloudQuantRef_pd])

    quant_vs_CloudQuantRef_data = quant_vs_CloudQuantRef_data.transpose()
    quant_vs_CloudQuantRef_data.columns = [f'{model_name} Quant', f'{model_name} Quant Ref']
    print(quant_vs_CloudQuantRef_data)
    quant_vs_CloudQuantRef_data = quant_vs_CloudQuantRef_data.stack().reset_index()
    quant_vs_CloudQuantRef_data.columns = ['Layer Index', 'Pipeline', 'normRMSE']

    print(quant_vs_CloudQuantRef_data)

    plt.figure(figsize=[5.2, 3.9])

    sns.lineplot(data=quant_vs_CloudQuantRef_data, x='Layer Index', y='normRMSE', hue="Pipeline", style="Pipeline",
                 markers=True)

    plt.legend([f'Quant vs Baseline', f'Quant Ref vs. Baseline'],
               loc='upper right')
    plt.xlabel("Layer Index")
    plt.ylabel("Normalized rMSE")
    plt.title(f"{model_name}")
    plt.ylim([0, 1])
    plt.savefig(quant_vs_CloudQuantRef_fig)

# def plot_all_in_one():
#     quant_vs_CloudQuantRef_fig = "python/plots/quant_vs_CloudQuantRef_rmse.png"
#
#     quant_vs_CloudQuantRef_data = pd.DataFrame(
#         [mobilenetv2_quant_vs_CloudQuantRef_pd, mobilenetv2_quantRef_vs_CloudQuantRef_pd,
#          mobilenetv3_quant_vs_CloudQuantRef_pd, mobilenetv3_quantRef_vs_CloudQuantRef_pd])
#
#     quant_vs_CloudQuantRef_data = quant_vs_CloudQuantRef_data.transpose()
#     quant_vs_CloudQuantRef_data.columns = ['MobileNet v2 Quant', 'MobileNet v2 Quant Ref', 'MobileNet v3 Quant',
#                                            'MobileNet v3 Quant Ref']
#     print(quant_vs_CloudQuantRef_data)
#     quant_vs_CloudQuantRef_data = quant_vs_CloudQuantRef_data.stack().reset_index()
#     quant_vs_CloudQuantRef_data.columns = ['Layer Index', 'Pipeline', 'normRMSE']
#
#     print(quant_vs_CloudQuantRef_data)
#
#     plt.figure(figsize=[5.6, 4.2])
#
#     sns.lineplot(data=quant_vs_CloudQuantRef_data, x='Layer Index', y='normRMSE', hue="Pipeline", style="Pipeline",
#                  markers=True)
#
#     plt.legend(['MobileNet v2 Quant', 'MobileNet v2 Quant Ref', 'MobileNet v3 Quant', 'MobileNet v3 Quant Ref'],
#                loc='upper right')
#     plt.xlabel("Layer Index")
#     plt.ylabel("Normalized rMSE")
#     plt.ylim([0, 0.8])
#     plt.savefig(quant_vs_CloudQuantRef_fig)


if __name__ == '__main__':
    # plot_refOp_vs_refTflite_rmse_fig()
    plot_quant_vs_CloudQuantRef_fig()

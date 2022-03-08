import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tqdm import tqdm
from MLEXray.ML_Diff.MLLogReader import MLLogReader
from MLEXray.Utils.params import ModelName, VectorLogKeys, DatasetName
from MLEXray.Utils.tf_utils import find_tensor_detail, unqunatize_tensor

IMAGE_SIZE = [224, 224, 3]


class MLDiffer:
    def __init__(self):
        self.mReader = MLLogReader()
        return

    def diff(self, trace_list, name_list,
             scalar_key_list=MLLogReader.scalar_key_list, vector_key_list=MLLogReader.vector_key_list,
             tflite_model_list=None,
             output_dir="./",
             per_layer_fig=False):
        """

        :param trace_list:
        :param name_list:
        :param scalar_key_list:
        :param vector_key_list:
        :param tflite_model_list:
        :param output_dir:
        :param per_layer_fig: set True to visualize each layers output
        :return:
        """

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        def data_type_conversion(data_frame):
            # print(data_frame.max(), data_frame.min())
            if data_frame.max() <= 1 and data_frame.min() < 0:
                # normalize from [-1,1] -> [0,1]
                data_frame = data_frame / 2 + 0.5
            elif data_frame.max() > 1:
                data_frame = data_frame.astype(int)
            return data_frame

        debug_report = ["layername, MSE, norm MSE\n"]
        # load the model tensor details if any
        tensor_detail_list = None
        if tflite_model_list is not None:
            debug_report = [
                "layer_loc,layername,MSE,MAE,max_m,min_m,max_n,min_n,norm MSE,norm MAE,norm_max_m,norm_min_m,norm_max_n,norm_min_n,normRMSE,max_abs_error\n"]
            tensor_detail_list = []
            for tflite_model_dir in tflite_model_list:
                interpreter = tf.lite.Interpreter(model_path=tflite_model_dir)
                tensor_details = interpreter.get_tensor_details()
                # print(tensor_details)
                tensor_detail_list.append(tensor_details)

        # visualize scalar difference
        if scalar_key_list is not None:
            for k in scalar_key_list:
                data = pd.concat([data_frame[k] for data_frame in trace_list], axis=1)
                data.columns = name_list
                print(data)
                print(data.mean())
                print(data.std())
                data = pd.melt(data, var_name='TraceID', value_name=k)
                if per_layer_fig:
                    plt.figure()
                    sns.barplot(x='TraceID', y=k, data=data)
                    plt.savefig(output_dir + k + ".png")
                    plt.close()

        # visualize vector difference
        if vector_key_list is None: return

        for k in vector_key_list:
            flag = False
            for data_frame in trace_list:
                if k not in data_frame:
                    # print(f"{k} doesn't exist")
                    flag = True
                    break
            if flag: continue
            data = pd.concat([data_frame[k] for data_frame in trace_list], axis=1)
            data.columns = name_list
            cols = len(name_list)
            for i in range(data.shape[0]):
                if k == VectorLogKeys.ModelInput:
                    plt.figure(figsize=(4 * cols, 4))
                    for j in range(cols):
                        data_frame = np.array(data[name_list[j]][i])
                        data_frame = data_type_conversion(data_frame)
                        image = np.reshape(data_frame, IMAGE_SIZE)
                        if per_layer_fig:
                            plt.subplot(1, cols, j + 1)
                            plt.imshow(image)  # assumes RGB channel
                            plt.title(name_list[j])
                    if per_layer_fig:
                        plt.savefig(output_dir + f"InputSBS_{i}.png")
                    plt.close()
                elif k == VectorLogKeys.TopKResults:
                    continue
                else:
                    # layer_name_list = [
                    #                     "Input",
                    #                     "Output",
                    #                     "Identity",
                    #                     # "Embeddings",
                    #                    "input_1",
                    #                    "model/Conv1_relu/Relu6;model/bn_Conv1/FusedBatchNormV3;model/Conv1/Conv2D",
                    #                    "model/Conv1/Conv2D",
                    #                    "model/conv1/relu/Relu;model/conv1/bn/FusedBatchNormV3;model/conv1/conv/Conv2D",
                    #                    "model/conv1/conv/Conv2D",
                    #                    ]
                    # if k not in layer_name_list: continue
                    raw_data = []
                    normalized_data = []
                    ratio = []
                    for j in range(cols):
                        data_frame = np.array(data[name_list[j]][i])
                        # normalize
                        # normalized_data_frame = (data_frame- min(data_frame)) / (max(data_frame) - min(data_frame))
                        tensor_detail = None
                        if tensor_detail_list is not None:
                            tensor_detail = find_tensor_detail(tensor_details=tensor_detail_list[j], tensor_name=k)
                        if tensor_detail is not None:
                            normalized_data_frame = unqunatize_tensor(data_frame, tensor_detail)
                        elif 130 > max(data_frame) > 100 or min(data_frame) < 0:
                            normalized_data_frame = (data_frame) / 128.0
                        elif max(data_frame) >= 130:
                            normalized_data_frame = (data_frame - 127) / 128.0
                        else:
                            normalized_data_frame = data_frame
                        data_frame = pd.DataFrame(data_frame)
                        normalized_data_frame = pd.DataFrame(normalized_data_frame)
                        raw_data.append(data_frame)
                        normalized_data.append(normalized_data_frame)

                    # calc MSE for raw data and MSE
                    # print(k)
                    layer_mse = k
                    tensor_detail_idx = None
                    if tensor_detail_list is not None:
                        tensor_detail = find_tensor_detail(tensor_details=tensor_detail_list[0], tensor_name=k)
                        tensor_detail_idx = tensor_detail['index']
                        layer_mse = f"{tensor_detail['index']} ,{k}"

                    for m in range(cols):
                        for n in range(m, cols):
                            if m == n: continue
                            mse = mean_squared_error(y_true=raw_data[m],
                                                     y_pred=raw_data[n])
                            mae = mean_absolute_error(y_true=raw_data[m],
                                                      y_pred=raw_data[n])
                            raw_max_m = raw_data[m].to_numpy().max()
                            raw_min_m = raw_data[m].to_numpy().min()

                            raw_max_n = raw_data[n].to_numpy().max()
                            raw_min_n = raw_data[n].to_numpy().min()

                            layer_mse += f", {mse}, {mae}, {raw_max_m}, {raw_min_m}, {raw_max_n}, {raw_min_n}"
                            # print(f"{name_list[m]} vs {name_list[n]} raw MSE: {mse}")
                            mse = mean_squared_error(y_true=normalized_data[m],
                                                     y_pred=normalized_data[n])
                            mae = mean_absolute_error(y_true=normalized_data[m],
                                                      y_pred=normalized_data[n])

                            norm_diff = normalized_data[m] - normalized_data[n]
                            # print(norm_diff)
                            max_abs_error = abs(norm_diff.to_numpy()).max()
                            # print(max_abs_error)

                            normalized_max_m = normalized_data[m].to_numpy().max()
                            normalized_min_m = normalized_data[m].to_numpy().min()

                            normalized_max_n = normalized_data[n].to_numpy().max()
                            normalized_min_n = normalized_data[n].to_numpy().min()

                            scale_m = normalized_max_m - normalized_min_m
                            scale_n = normalized_max_n - normalized_min_n
                            scale = max(scale_m, scale_n)


                            normRMSE = mse ** (1 / 2) / scale

                            layer_mse += f",{mse}, {mae}, {normalized_max_m}, {normalized_min_m}, {normalized_max_n}, {normalized_min_n}, {normRMSE}, {max_abs_error}\n"
                            # layer_mse += f", {mse}, {mae}\n"
                            # print(f"{name_list[m]} vs {name_list[n]} normalized MSE: {mse}")

                    debug_report.append(layer_mse)
                    # visualize
                    if per_layer_fig:
                        visualize(raw_data, normalized_data, name_list, output_dir,
                                  data_key=k, data_index=i, data_key_index=tensor_detail_idx)

        f = open(f"{output_dir}/0_0_debug_report.csv", "w")
        f.writelines(debug_report)
        f.close()


def visualize(raw_data, normalized_data, name_list, output_dir, data_key, data_index, data_key_index=None):
    # visualize
    if len(data_key) > 20:
        data_key = data_key[:20]
    data_key = data_key.replace('/', '_')

    normalized_data = pd.concat(normalized_data, axis=1)
    raw_data = pd.concat(raw_data, axis=1)
    normalized_data.columns = name_list
    raw_data.columns = name_list
    normalized_data = normalized_data[0:1000]
    raw_data = raw_data[0:1000]
    plt.figure()
    sns.scatterplot(data=normalized_data)
    plt.ylabel("Normalized " + data_key)
    plt.xlabel("Index")
    plt.savefig(f"{output_dir}{data_key_index}_{data_key}_{data_index}_Normalized.png")
    plt.close()
    plt.figure()
    sns.scatterplot(data=raw_data)
    plt.ylabel(data_key)
    plt.xlabel("Index")
    plt.savefig(f"{output_dir}{data_key_index}_{data_key}_{data_index}.png")
    plt.close()
    for p in raw_data.columns:
        plt.figure()
        sns.scatterplot(data=raw_data[p])
        plt.ylabel(data_key)
        plt.xlabel("Index")
        plt.savefig(f"{output_dir}{data_key_index}_{data_key}_{data_index}_{p}.png")
        plt.close()


def test_ml_diff():
    mDiffer = MLDiffer()

    output_dir = "./analysis/"
    # trace_name = "Sat_Aug_07_16-19-05_PDT_2021_Classification"
    trace_name = DatasetName.Imagenet_100
    model_name = ModelName.MobileNetV2

    data_trace = f"data/0_data/{trace_name}"
    # cloud_trace = f"data/1_cloud/{trace_name}"
    # tflite_trace = "data/2_tflite/Sat_Aug_07_16-19-05_PDT_2021_Classification_playback_at_Sat_Aug_07_16-39-04_PDT_2021_classification"
    # tflite_quant_trace = "data/3_tflite_quant/Sat_Aug_07_16-19-05_PDT_2021_Classification_playback_at_Sat_Aug_07_20-44-15_PDT_2021_classification"
    cloud_trace = "data/trace_imagenet2012_100/MobileNetV2/1_cloud/imagenet2012_100"
    # tflite_trace = "data/trace_imagenet2012_100/MobileNetV2/2_tflite/imagenet2012_100_playback_at_Mon_Aug_09_01-18-46_PDT_2021_classification"
    # tflite_quant_trace = "data/trace_imagenet2012_100/MobileNetV2/3_tflite_quant/imagenet2012_100_playback_at_Tue_Aug_24_18-09-05_PDT_2021_classification"
    # name_list = ["MobileNetV2_Cloud_CKPT", "MobileNetV2_TFLite", "MobileNetV2_TFLite_Quant"]
    name_list = ["MobileNetV2_TFLite", "MobileNetV2_TFLite_Quant"]

    # pixel 4
    # tflite_trace = "data/Benchmark/mobilenetv2/pixel4/4_thread/0_None_imagenet2012_10_playback_at_Wed_Oct_06_21-48-18_PDT_2021_classification"
    # tflite_quant_trace = "data/Benchmark/mobilenetv2/pixel4/4_thread/1_IO_imagenet2012_10_playback_at_Wed_Oct_06_21-48-24_PDT_2021_classification"

    # tflite_trace = "data/Benchmark/mobilenetv2/pixel4/gpu_4_threads/0_None_imagenet2012_10_playback_at_Wed_Oct_06_22-38-04_PDT_2021_classification"
    # tflite_quant_trace = "data/Benchmark/mobilenetv2/pixel4/gpu_4_threads/1_IO_imagenet2012_10_playback_at_Wed_Oct_06_22-38-12_PDT_2021_classification"

    # pixel 3
    # tflite_trace = "data/Benchmark/mobilenetv2/pixel3/cpu_none/imagenet2012_10_playback_at_Mon_Oct_11_15-07-59_PDT_2021_classification"
    # tflite_quant_trace = "data/Benchmark/mobilenetv2/pixel3/cpu_io/imagenet2012_10_playback_at_Mon_Oct_11_15-10-55_PDT_2021_classification"

    # tflite_trace = "data/Benchmark/mobilenetv2/pixel3/gpu_io/imagenet2012_10_playback_at_Mon_Oct_11_15-13-02_PDT_2021_classification"
    # tflite_quant_trace = "data/Benchmark/mobilenetv2/pixel3/gpu_none/imagenet2012_10_playback_at_Mon_Oct_11_15-14-49_PDT_2021_classification"
    tflite_trace = "data/Benchmark/mobilenetv2/pixel3/gpu_io/imagenet2012_10_playback_at_Mon_Oct_11_15-13-22_PDT_2021_classification"
    tflite_quant_trace = "data/Benchmark/mobilenetv2/pixel3/gpu_none/imagenet2012_10_playback_at_Mon_Oct_11_15-15-16_PDT_2021_classification"

    name_list = ["Baseline", "Instrumented"]

    # scalar_key_list = ['Inference Latency']
    scalar_key_list = ['Inference Latency', 'FPS Latency']
    vector_key_list = ['Embeddings', 'Input']

    mReader = MLLogReader(addtional_vector_list=vector_key_list, addtional_scalar_list=scalar_key_list)

    pd_data = mReader.read_log(data_trace)
    pd_cloud = mReader.read_log(cloud_trace)
    pd_tflite = mReader.read_log(tflite_trace)
    pd_tflite_quant = mReader.read_log(tflite_quant_trace)

    print(pd_cloud.keys())
    print(pd_tflite.keys())
    print(pd_tflite_quant.keys())

    trace_list = [pd_tflite, pd_tflite_quant]

    mDiffer.diff(trace_list, name_list, scalar_key_list=scalar_key_list,
                 vector_key_list=vector_key_list, output_dir=output_dir)


def debug_quantization_per_layer_output():
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


if __name__ == "__main__":
    # test_ml_diff()
    debug_quantization_per_layer_output()

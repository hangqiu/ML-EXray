import os
from tqdm import tqdm
from MLEXray.ML_Diff.MLLogReader import MLLogReader
from MLEXray.Utils.params import ModelName, ScalarLogKeys, VectorLogKeys


class TopKAccuracy(object):
    num_inference = 0
    gt = []
    inf_results = []
    model_output_label = []
    def __init__(self, groundtruth_filepath, model_output_label, num_samples=100):
        """

        :param pd_data: this is the inference result column from the log
        :param groundtruth_filepath:
        """
        # read groundtruth
        self.parse_model_output_label_file(model_output_label)
        self.parse_gt_file(groundtruth_filepath, num_samples=num_samples)

    def __del__(self):
        self.gt = []
        self.inf_results = []
        self.model_output_label = []

    def parse_model_output_label_file(self, model_output_label):
        try:
            f = open(model_output_label)
        except Exception as e:
            raise e
        self.model_output_label = f.readlines()
        f.close()
        print(f"Loaded model output {len(self.model_output_label)}")


    def parse_gt_file(self, groundtruth_filepath, num_samples):
        try:
            f = open(groundtruth_filepath)
        except Exception as e:
            raise e

        lines = f.readlines()
        f.close()
        if num_samples > len(lines):
            raise ValueError(f"Groundtruth ({len(lines)}) less than for inference results ({num_samples})")

        for i in range(num_samples):
            self.gt.append(lines[i])

        self.num_inference = len(self.gt)
        print(f"Loaded Ground Truth {self.num_inference} records")
        # print(self.gt)



    def parse_inference_result(self, pd_data):
        self.inf_results = []
        for single_result in pd_data:
            idx = 0
            inf_res = []
            while idx < len(single_result):
                res_label = self.model_output_label[int(single_result[idx])]
                inf_res.append(res_label)
                idx += 2
            self.inf_results.append(inf_res)

        print("Loaded Inference Results")
        # print(self.inf_results)

    def eval_accuracy(self, topk=1):
        accuracy = []
        for i in range(topk):
            acc = self.eval_topK_accuracy(i+1)
            accuracy.append(acc)
        return accuracy

    def eval_topK_accuracy(self, topk=1):
        num_correct = 0
        assert len(self.inf_results) == self.num_inference
        for i in range(self.num_inference):
            results = self.inf_results[i][0:topk]
            if self.gt[i] in results:
                num_correct += 1
        accuracy = float(num_correct) / float(self.num_inference)
        return accuracy

def get_log_path(log_dir):
    """from a folder, load all log paths in that folder"""
    ret = []
    log_paths = os.listdir(log_dir)
    for runs in log_paths:
        run_dir = log_dir + '/' + runs
        for run in os.listdir(run_dir):
            run_path = log_dir + '/' + runs + '/' + run
            print(run_path)
            ret.append(run_path)
    return ret

if __name__ == '__main__':
    groundtruth_filepath = "data/0_data/imagenet2012/groundtruth_label.txt"
    model_output_label = "model/imagenet_label_1000.txt"
    # model_output_label = "model/imagenet_label_1001.txt"

    trace_name = "imagenet2012_100"

    # model_name = ModelName.MobileNetV1
    # model_name = ModelName.MobileNetV2
    # model_name = ModelName.MobileNetV3_Large
    # model_name = ModelName.MobileNetV3_Small
    # model_name = ModelName.NASNetMobile
    # model_name = ModelName.InceptionV3
    model_name = ModelName.DenseNet121
    # model_name = ModelName.ResNet50V1
    # model_name = ModelName.ResNet50V2

    # log_path = f"data/{model_name}/1_cloud/{trace_name}"
    # log_path = f"data/{model_name}/2_tflite/imagenet2012_100_playback_at_Mon_Aug_23_22-25-10_PDT_2021_classification"
    # log_path = f"data/{model_name}/2_tflite_A_BGR/imagenet2012_100_playback_at_Mon_Aug_09_01-22-38_PDT_2021_classification"
    # log_path = f"data/{model_name}/2_tflite_B_bilinear/imagenet2012_100_playback_at_Mon_Aug_09_01-24-49_PDT_2021_classification"
    # log_path = f"data/{model_name}/2_tflite_C_[0,1]/imagenet2012_100_playback_at_Mon_Aug_09_01-26-54_PDT_2021_classification"
    # log_path = f"data/{model_name}/2_tflite_D_Rot90/imagenet2012_100_playback_at_Tue_Aug_31_11-42-04_PDT_2021_classification"

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

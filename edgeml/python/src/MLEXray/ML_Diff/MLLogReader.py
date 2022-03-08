import os
import pandas as pd
import tensorflow as tf

class MLLogReader(object):
    scalar_key_list = []
    vector_key_list = []
    debug = False

    def __init__(self, addtional_scalar_list=None, addtional_vector_list=None, model_path=None, per_layer_output=False):
        if per_layer_output and (model_path is not None):
            interpreter = tf.lite.Interpreter(model_path=model_path)
            tensor_details = interpreter.get_tensor_details()
            for tensor in tensor_details:
                if 'name' in tensor:
                    tensor_name = tensor['name']
                    if tensor_name != "":
                        self.vector_key_list.append(tensor_name)
        if addtional_scalar_list is not None:
            self.scalar_key_list.extend(addtional_scalar_list)
        if addtional_vector_list is not None:
            self.vector_key_list.extend(addtional_vector_list)
        if self.debug:
            print(f"Initialized MLLog Reader with scalar keys: {self.scalar_key_list}")
            print(f"Initialized MLLog Reader with vector keys: {self.vector_key_list}")
        return

    def read_log(self, logDir):
        if not os.path.exists(logDir):
            raise ValueError(f"{logDir} doesn't exist")
        native_pd_data = self.read_native_log(logDir + '/nativeLog/')
        java_pd_data = self.read_java_log(logDir + '/log/')
        pd_data = pd.concat([native_pd_data, java_pd_data], axis=1)
        return pd_data

    def read_log_file(self, file_path, frameId, data):
        if self.debug: print(f"Reading file {file_path}")
        f = open(file_path, 'r')
        while f:
            line = f.readline()
            if line == "":
                break
            kv = line.split(':')
            if len(kv) < 2:
                continue
            k = kv[0].strip()
            if self.debug: print(f"Reading key {k}")
            v = kv[1].strip().strip('[').strip(']').strip('ms')
            if v == "":
                if self.debug: print("empty value")
                continue
            if k in self.scalar_key_list:
                if not k in data:
                    data[k] = dict()
                data[k][frameId] = float(v)
            elif k in self.vector_key_list:
                if not k in data:
                    data[k] = dict()
                if ',' in v:
                    v = [float(i) for i in v.split(',')]
                else:
                    v = [float(i) for i in v.split(' ')]
                data[k][frameId] = v
            else:
                if self.debug: print(f"{k} not in logging key list")
        f.close()
        if self.debug: print(data.keys())

    def read_native_log(self, native_log_path):
        frameId = 0
        data = dict()
        frameId_str = '%08d' % frameId
        while os.path.exists(native_log_path + frameId_str):
            if self.debug:
                print(f"Reading {native_log_path + frameId_str}")
            self.read_log_file(native_log_path + frameId_str, frameId, data)
            frameId += 1
            frameId_str = '%08d' % frameId

        pd_data = pd.DataFrame(data)

        return pd_data

    def read_java_log(self, java_log_path):

        frameId = 0
        data = dict()
        while os.path.exists(java_log_path + str(frameId) + '.log'):
            self.read_log_file(java_log_path + str(frameId) + '.log', frameId, data)
            frameId += 1

        pd_data = pd.DataFrame(data)

        return pd_data


# unit test
if __name__ == "__main__":
    mReader = MLLogReader()

    # test mobile logs
    log_path = "data/0_data/Sat_Aug_07_16-19-05_PDT_2021_Classification"
    # test cloud logs
    log_path = "data/1_cloud/Sat_Aug_07_16-19-05_PDT_2021_Classification"
    # test cloud converted tflite logs
    log_path = "data/2_tflite/Sat_Aug_07_16-19-05_PDT_2021_Classification_playback_at_Sat_Aug_07_16-39-04_PDT_2021_classification/"
    # test cloud converted tflite logs
    log_path = "data/3_tflite_quant/Sat_Aug_07_16-19-05_PDT_2021_Classification_playback_at_Sat_Aug_07_20-44-15_PDT_2021_classification"

    native_log_path = log_path + "nativeLog/"
    java_log_path = log_path + "log/"

    # print(mReader.read_native_log(native_log_path))
    # print(mReader.read_java_log(java_log_path))
    print(mReader.read_log(log_path))

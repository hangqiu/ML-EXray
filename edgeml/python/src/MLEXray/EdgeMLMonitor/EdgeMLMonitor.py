import os
import time

class EdgeMLMonitor():
    logDir = ""
    logID = 0
    inferenceLogID = 0

    isInferenceLogging = False

    inferenceLatencyStart = 0
    inferenceLatencyStop = 0
    infernceLatency_ms = 0

    frameLatencyStart = 0
    frameLatencyStop = 0
    frameLatency_ms = 0

    inferenceResultString = ""
    nativeLogString = ""
    mInferenceResults = ""

    per_layer_output = False

    def set_per_layer_output_logging(self, value):
        self.per_layer_output = value

    def set_logDir(self, logDIr=None):
        self.logDir = logDIr
        if not os.path.exists(self.logDir + "/log/"):
            os.makedirs(self.logDir + "/log/")
        if not os.path.exists(self.logDir + "/nativeLog/"):
            os.makedirs(self.logDir + "/nativeLog/")

    def onFrameStart(self):
        self.frameLatencyStart = time.time()

    def onFrameStop(self):
        self.frameLatencyStop = time.time()
        self.frameLatency_ms = self.frameLatencyStop - self.frameLatencyStart

    def onInferenceStart(self):
        if self.isInferenceLogging: return
        self.isInferenceLogging = True
        self.inferenceLatencyStop = 0
        self.infernceLatency_ms = 0
        self.mInferenceResults = ""
        self.mEmbeddings = ""
        self.mInput = ""
        self.mOutput = ""
        self.mRawInput = ""
        self.layer_output = dict()
        # print("onInferenceStart!")
        self.inferenceLatencyStart = time.time()

    def onInferenceStop(self, interpreter, result=None, raw_input=None):
        """

        :param result: A dict of inference results
            - predictions: final output vector
            - embeddings [optional]: last layer before activation
            - input[optional]:
        :param raw_input: input tensor
        :return:
        """
        self.inferenceLatencyStop = time.time()
        if result is not None:
            self.mOutput = result['predictions'].flatten().tolist()
            self.mInferenceResults = self.decode_classification_results(result['predictions'])
            if 'embedding' in result:
                self.mEmbeddings = result['embedding'].flatten().tolist()
            if 'input' in result:
                self.mInput = result['input'].flatten().tolist()

        if raw_input is not None:
            self.mRawInput = raw_input.numpy().flatten().tolist()

        if self.per_layer_output:
            per_layer_result = dict()
            tensor_details = interpreter.get_tensor_details()
            for tensor in tensor_details:
                if tensor['name'] != "":
                    value = interpreter.get_tensor(tensor['index'])
                    per_layer_result[tensor['name']] = value
            self.layer_output = per_layer_result

        self.inferenceResultString = self.getInferenceLogString()
        self.nativeLogString = self.getNativeLogString()
        if not self.isInferenceLogging:
            return
        # print("onInferenceStop!")
        self.isInferenceLogging = False
        self.appendInferenceLog()

    def getInferenceLogString(self):
        self.infernceLatency_ms = self.inferenceLatencyStop - self.inferenceLatencyStart
        return f"Inference Start Time: {self.inferenceLatencyStart} ms\n" \
               f"Inference Time: {self.infernceLatency_ms} ms\n" \
               f"Inference Result: {self.mInferenceResults}\n"

    def decode_classification_results(self, result, top=10):
        ret = []
        for r in result:
            top_indices = r.argsort()[-top:][::-1]
            rr = []
            for i in range(top):
                rr.append(top_indices[i])
                rr.append(r[top_indices[i]])
            ret.append(rr)
        return ret

    def getNativeLogString(self):
        layer_output_string = ""
        if self.per_layer_output:
            for k in self.layer_output:
                if k in ['predictions', 'embedding', 'input']:continue
                v = self.layer_output[k].flatten().tolist()
                layer_output_string += f"{k}: {v}\n"

        return f"Inference Latency: {self.infernceLatency_ms}ms\n" \
               f"Layer Output:\n{layer_output_string}\n" \
               f"Output: {self.mOutput} \n" \
               f"Embeddings: {self.mEmbeddings} \n" \
               f"Input: {self.mInput}\n" \
               f"RawInput: {self.mRawInput}\n"



    def appendInferenceLog(self):
        self.saveInferenceLog()
        self.saveNativeLog()
        self.inferenceLogID += 1

    def saveInferenceLog(self):
        filepath = f"{self.logDir}/log/{self.inferenceLogID}.log"
        f = open(filepath, 'w+')
        f.writelines(self.inferenceResultString)
        f.close()

    def saveNativeLog(self):
        log_name = '%08d' % self.inferenceLogID
        filepath = f"{self.logDir}/nativeLog/{log_name}"
        f = open(filepath, 'w+')
        f.writelines(self.nativeLogString)
        self.onFrameStop()
        f.writelines(f"FPS Latency: {self.frameLatency_ms}\n")
        f.close()
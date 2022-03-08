from MLEXray.EdgeMLMonitor.EdgeMLMonitor import EdgeMLMonitor
import tensorflow as tf

class Model_Runner():
    input_size = None
    def __init__(self, model_name):
        self.model_name = model_name
        self.mMonitor = EdgeMLMonitor()

    def invoke_model(self, input):
        pass


    def init_tflite(self,
                 model_name,
                 quantized,
                 tflite_model_dir=None,
                 tflite_model=None,
                 per_layer_logging=False,
                 referenceOp=False
                 ):
        if tflite_model is None and tflite_model_dir is None:
            raise ValueError(f"Either model content or model path has to be provided")
        OpResolver = tf.lite.experimental.OpResolverType.AUTO
        if referenceOp:
            OpResolver = tf.lite.experimental.OpResolverType.BUILTIN_REF
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model,
                                               model_path=tflite_model_dir,
                                               experimental_preserve_all_tensors=True,
                                               experimental_op_resolver_type=OpResolver)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        print("Input Details", self.input_details)
        print("Output Details", self.output_details)
        self.input_size = self.input_details['shape'][-2]
        self.quantized = quantized
        print("tflite model built")

        self.__init__(model_name)
        if per_layer_logging:
            self.mMonitor.set_per_layer_output_logging(True)
        else:
            self.mMonitor.set_per_layer_output_logging(False)

    def set_per_layer_logging(self, value):
        self.mMonitor.set_per_layer_output_logging(value)

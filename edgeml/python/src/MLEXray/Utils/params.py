class ModelName(enumerate):
    '''
    Model names from tf.keras.applications, using version 2.6.0
    https://www.tensorflow.org/api_docs/python/tf/keras/applications
    '''
    # image classification
    MobileNetV1 = 'MobileNetV1'
    MobileNetV2 = 'MobileNetV2'
    MobileNetV3_Large = 'MobileNetV3_Large'
    MobileNetV3_Small = 'MobileNetV3_Small'
    InceptionV3 = 'InceptionV3'
    DenseNet121 = 'DenseNet121'
    EfficientNetB0 = 'EfficientNetB0'
    ResNet50V1 = 'ResNet50V1'
    ResNet50V2 = 'ResNet50V2'
    NASNetMobile = 'NASNetMobile'
    # detection
    SSDMobileNetv2 = 'SSDMobileNetv2'
    EfficientDet_D0 = 'EfficientDet_D0'
    FasterRCNNResnet101v1 = "FasterRCNN\nResnet101v1"

    # audio
    SpeechCommandModel = "SpeechCommandModel"
    YamNet = "YamNet"
    # text
    TextClassifierModel = "TextClassifierModel"

    @staticmethod
    def get_tflite_model_path(model_name, quantized=False):
        if not quantized:
            tflite_path = f"model/ConvertedModels/{model_name}/{model_name}_imagenet_224.tflite"
            if model_name == ModelName.InceptionV3:
                tflite_path = f"model/ConvertedModels/{model_name}/{model_name}_imagenet_299.tflite"
        else:
            tflite_path = f"model/ConvertedModels/{model_name}/{model_name}_imagenet_quant_224.tflite"
            if model_name == ModelName.InceptionV3:
                tflite_path = f"model/ConvertedModels/{model_name}/{model_name}_imagenet_quant_299.tflite"
        return tflite_path


class DatasetName(enumerate):
    # image
    Imagenet = "imagenet2012"
    Imagenet_1 = "imagenet2012_1"
    Imagenet_10 = "imagenet2012_10"
    Imagenet_100 = "imagenet2012_100"
    # audio
    AudioSet = "audioset"
    AudioSet_1 = "audioset_1"
    AudioSet_10 = "audioset_10"
    # text

    # detection
    COCO2014 = 'coco2014'
    COCO2014_100 = 'coco2014_100'
    COCO2017 = 'coco2017'
    COCO2017val_100 = 'coco2017_val_100'
    COCO2017val_300 = 'coco2017_val_300'

class ScalarLogKeys(enumerate):
    InferenceLatency = 'Inference Latency'
    MemoryCopyLatency = 'MemCopy Latency'
    ResizingLatency = 'Resizing Latency'


class VectorLogKeys(enumerate):
    ModelInput = "Input"
    ModelOutput = "Output"
    ModelEmbedding = "Embeddings"
    TopKResults = 'Inference Result'


class PipelineName(enumerate):
    Cloud = "Cloud"
    Reference = "Reference"
    RefQuant = "Mobile Quant Ref"
    Mobile = "Mobile"
    MobileQuant = "Mobile Quant"
    # preprocessing buggy pipelines
    MobileResize = "Resize"
    MobileChannel = "Channel"
    MobileNormalization = "Normalization"
    MobileRotation = "Rotation"

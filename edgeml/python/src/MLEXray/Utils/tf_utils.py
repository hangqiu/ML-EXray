import tensorflow as tf
import numpy as np
from MLEXray.Utils.params import ModelName



def KerasModel_preprocessing(image, model_name=None):
    if model_name == ModelName.InceptionV3:
        image = tf.keras.applications.inception_v3.preprocess_input(image)
    elif model_name in [ModelName.ResNet50V1]:
        image = tf.keras.applications.resnet50.preprocess_input(image)
    elif model_name in [ModelName.ResNet50V2]:
        image = tf.keras.applications.resnet_v2.preprocess_input(image)
    elif model_name == ModelName.DenseNet121:
        # image = tf.keras.applications.densenet.preprocess_input(image)
        image = tf.keras.applications.mobilenet.preprocess_input(image)
    elif model_name in [ModelName.MobileNetV3_Small, ModelName.MobileNetV3_Large]:
        # use mobilenet preprocessing function, cuz our keras model does not include preprocessing
        image = tf.keras.applications.mobilenet.preprocess_input(image)
    elif model_name == ModelName.MobileNetV2:
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    elif model_name == ModelName.MobileNetV1:
        image = tf.keras.applications.mobilenet.preprocess_input(image)
    else:
        image = tf.cast(image, tf.uint8)
        # raise ValueError(f"{model_name} not supported in preprocessing")
    return image

def quantize_tensor(tensor, tensor_details):
    """
    scale the tensor according to the specified detail from tflite model
    :param tensor:
    :param tensor_details:
    :return:
    """
    scale, zero_point = tensor_details['quantization']
    # print(f"tensor_details: name:{tensor_details['name']}, scale:{scale}, zero_point:{zero_point}, dtype:{tensor_details['dtype']}")
    integer_tensor = tensor / scale + zero_point
    integer_tensor = integer_tensor.astype(tensor_details['dtype'])
    return integer_tensor

def unqunatize_tensor(tensor, tensor_details):
    """
    scale the integer tensor back to float according to quantization details
    :param tensor:
    :param tensor_details:
    :return:
    """
    scale, zero_point = tensor_details['quantization']
    # print(f"tensor_details: name:{tensor_details['name']}, scale:{scale}, zero_point:{zero_point}, dtype:{tensor_details['dtype']}")
    if scale == 0.0:
        return tensor
    float_tensor = tensor.astype(np.float32)
    float_tensor = (float_tensor - zero_point) * scale
    return float_tensor

def find_tensor_detail(tensor_details, tensor_name):
    for tensor in tensor_details:
        if tensor_name == tensor['name']:
            return tensor
    return None
import os

import cv2
import numpy as np
import tensorflow as tf
from MLEXray.Model_Runner import Keras_Image_Model_Runner
from MLEXray.Utils.params import ModelName
from MLEXray.Utils.tf_utils import KerasModel_preprocessing
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow_datasets as tfds
print(tf.__version__)
print(tfds.__version__)

class Model_Converter():

  @staticmethod
  def SavedModelDir2Tflite(saved_model_dir, model_name, output_model_dir, quantize=None):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
    return Model_Converter.to_tflite(converter=converter, model_name=model_name, output_model_dir=output_model_dir, input_size=None, quantize=quantize)

  @staticmethod
  def KerasModelDir2Tflite(saved_model_dir, model_name, save_model_dir, quantize=None):
    model = tf.keras.models.load_model(saved_model_dir)
    return Model_Converter.KerasModel2Tflite(model, model_name=model_name, save_model_dir=save_model_dir, quantize=quantize)

  @staticmethod
  def KerasModel2Tflite(keras_model, model_name, save_model_dir, input_size=None, quantize=None):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    return Model_Converter.to_tflite(converter=converter, model_name=model_name, output_model_dir=save_model_dir, input_size=input_size, quantize=quantize)

  @staticmethod
  def to_tflite(converter, model_name, output_model_dir, input_size=None, quantize=None):
    if quantize is not None:
      # Full integer quantization only
      # Uint8 quant only for now, consistent with tf model zoo
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
      converter.representative_dataset = get_representative_dataset(input_size=input_size, model_name=model_name)
      converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
      # converter.target_spec.supported_types = [tf.int8]
      converter.inference_input_type = tf.uint8  # tf.uint8 or tf.int8
      converter.inference_output_type = tf.uint8  # tf.uint8python or tf.int8

    print('Starting conversion')
    tflite_model = converter.convert()
    print('Ending conversion')
    # Save the model.
    with open(output_model_dir + '.tflite', 'wb') as f:
      f.write(tflite_model)
    return tflite_model

def get_representative_dataset(input_size=None, model_name=None):
  if input_size is None:
    input_size = 224
  sample_size = 100
  dataset = tfds.load('imagenet_v2', split='test', shuffle_files=True)

  def representative_dataset():
    for data in dataset.batch(1).take(sample_size):
      image = data['image']
      # plt.figure()
      # plt.imshow(image[0])
      # plt.savefig("tmp.png")
      # plt.close()
      # print(image)
      image = tf.cast(image, tf.float32)
      image = KerasModel_preprocessing(image, model_name=model_name)
      image = tf.image.resize(image, [input_size, input_size], method=tf.image.ResizeMethod.AREA)
      # image = tf.cast(image, tf.uint8)
      # print(image)
      # plt.figure()
      # plt.imshow(image[0])
      # plt.savefig("tmp_0.png")
      # plt.close()

      yield [image]

  return representative_dataset


def MobileNetV1_SavedModel():
  saved_model_dir = "model/SavedModels_TFHub/imagenet_mobilenet_v1_100_224_classification_5/"
  model_name_dir = saved_model_dir + "mobilenet_v1_100_224_imagenet_224"
  Model_Converter.SavedModelDir2Tflite(saved_model_dir=saved_model_dir, model_name=ModelName.MobileNetV1, output_model_dir=model_name_dir)
  model_name_dir = saved_model_dir + "mobilenet_v1_100_224_imagenet_quant_224"
  Model_Converter.SavedModelDir2Tflite(saved_model_dir=saved_model_dir, model_name=ModelName.MobileNetV1, output_model_dir=model_name_dir, quantize=True)

def MobileNetV2_SavedModel():
  saved_model_dir = "model/SavedModels_TFHub/imagenet_mobilenet_v2_100_224_classification_5/"
  model_name_dir = saved_model_dir + "mobilenet_v2_100_224_imagenet_224"
  Model_Converter.SavedModelDir2Tflite(saved_model_dir=saved_model_dir, model_name=ModelName.MobileNetV2, output_model_dir=model_name_dir)
  model_name_dir = saved_model_dir + "mobilenet_v2_100_224_imagenet_quant_224"
  Model_Converter.SavedModelDir2Tflite(saved_model_dir=saved_model_dir, model_name=ModelName.MobileNetV2, output_model_dir=model_name_dir, quantize=True)

def MobileNetV2_Keras():
  saved_model_dir = "model/ConvertedModels/imagenet_mobilenet_v2_100_224_converted_tflite/"
  if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
  model = tf.keras.applications.mobilenet_v2.MobileNetV2()
  model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
  )
  model.summary()
  model_name_dir = saved_model_dir + "mobilenet_v2_100_imagenet_224"
  Model_Converter.KerasModel2Tflite(model, model_name=ModelName.MobileNetV2,  save_model_dir=model_name_dir)

  model_name_dir = saved_model_dir + "mobilenet_v2_100_imagenet_quant_224"
  tflite_quantize_model = Model_Converter.KerasModel2Tflite(model, model_name=ModelName.MobileNetV2,  save_model_dir=model_name_dir, quantize=True)
  check_model_output(model, tflite_quantize_model)


def check_model_output(keras_model=None, model_name=None, input_size=224, tflite_quantize_model=None, tflite_quantize_model_dir=None):
  # Input to the TF model are float values in the range [0, 10] and of size (1, 100)
  # np.random.seed(0)
  # tf_input = np.random.uniform(low=-1, high=1, size=(1, 224, 224, 3)).astype(np.float32)

  # file = tf.keras.utils.get_file(
  #   "grace_hopper.jpg",
  #   "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
  # img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])

  file = "data/0_data/imagenet2012_10/nativeInput/ILSVRC2012_val_00000003.JPEG"
  img = cv2.imread(file)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = [img]

  # dataset = tfds.load('imagenet_v2', split='test', shuffle_files=True)
  # for data in dataset.batch(1).take(1):
  #   img = data['image']

  plt.figure()
  plt.imshow(img[0])
  plt.axis('off')
  plt.savefig('tmp.png')
  print(img[0][0][0][0])

  # img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
  img = tf.image.resize(img, [input_size, input_size], method=tf.image.ResizeMethod.AREA)
  img = img[0].numpy()

  img = tf.keras.preprocessing.image.img_to_array(img)
  tf_input = KerasModel_preprocessing(img[tf.newaxis, ...], model_name=model_name)

  plt.figure()
  plt.imshow(tf_input[0])
  plt.axis('off')
  plt.savefig('tmp_0.png')
  print(tf_input[0][0][0][0])

  # Output of the TF model.
  if keras_model is not None:
    tf_output = keras_model.predict(tf_input)['predictions']
    tf_top_label_index = np.argmax(tf_output)
    tf_top_score = tf_output.flatten()[tf_top_label_index]
    print("tf: label index = %d, score = %f" % (tf_top_label_index, tf_top_score))

  # Output of the TFLite model.
  if tflite_quantize_model is not None or tflite_quantize_model_dir is not None:
    interpreter = tf.lite.Interpreter(model_content=tflite_quantize_model, model_path=tflite_quantize_model_dir)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    # Manually quantize the input from float to integer
    scale, zero_point = input_details['quantization']
    print(f"Input details: scale:{scale}, zero_point:{zero_point}, dtype:{input_details['dtype']}")
    tflite_integer_input = tf_input / scale + zero_point
    tflite_integer_input = tflite_integer_input.astype(input_details['dtype'])
    # print(tflite_integer_input.shape)
    interpreter.set_tensor(input_details['index'], tflite_integer_input)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    tflite_integer_output = interpreter.get_tensor(output_details['index'])
    # print(tflite_integer_output)
    # Manually dequantize the output from integer to float
    scale, zero_point = output_details['quantization']
    print(f"Output details: scale:{scale}, zero_point:{zero_point}")
    tflite_output = tflite_integer_output.astype(np.float32)
    tflite_output = (tflite_output - zero_point) * scale

    # check output prediction
    tflite_top_label_index = np.argmax(tflite_output)
    tflite_top_score = tflite_output.flatten()[tflite_top_label_index]

    print("tflite: label index = %d, score = %f" % (tflite_top_label_index, tflite_top_score))


  # print(tf_output.shape)
  # print(tf_output[0][0])
  # print(tflite_output.shape)
  # print(tflite_output[0][0])
  # Verify that the TFLite model's output is approximately (expect some loss in
  # accuracy due to quantization) the same as the TF model's output
  # assert np.allclose(tflite_output, tf_output, atol=1e-04) == True


def ConvertKerasModel(model_name, dataset_name='imagenet'):
  saved_model_dir = f"model/ConvertedModels/{model_name}/"
  if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
  model_runner = Keras_Image_Model_Runner.Keras_Image_Model_Runner(model_name=model_name)

  model = model_runner.model
  input_shape = model.input_shape
  print(input_shape)
  input_size = input_shape['input'][-2]
  print(input_size)
  model_name_dir = saved_model_dir + f"{model_name}_{dataset_name}_{input_size}"
  Model_Converter.KerasModel2Tflite(model, model_name=model_name, save_model_dir=model_name_dir, input_size=input_size)

  model_name_dir = saved_model_dir + f"{model_name}_{dataset_name}_quant_{input_size}"
  tflite_quantize_model = Model_Converter.KerasModel2Tflite(model, model_name=model_name, save_model_dir=model_name_dir, input_size=input_size, quantize=True)
  check_model_output(model, model_name=model_name, input_size=input_size, tflite_quantize_model=tflite_quantize_model)


def CheckKerasModelOutput(model_name=None, dataset_name='imagenet'):
  saved_model_dir = f"model/ConvertedModels/{model_name}/"
  if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)

  model = Keras_Image_Model_Runner.Keras_Image_Model_Runner(model_name=model_name).model
  input_shape = model.input_shape
  input_size = input_shape['input'][-2]

  model_name_dir = saved_model_dir + f"{model_name}_{dataset_name}_quant_{input_size}.tflite"
  print(model_name_dir)
  check_model_output(keras_model=model, model_name=model_name, input_size=input_size, tflite_quantize_model_dir=model_name_dir)

if __name__ == '__main__':
  # MobileNetV2_SavedModel()
  # MobileNetV2_Keras()
  # MobileNetV1_SavedModel()

  # model_name_list = [ModelName.MobileNetV1, ModelName.MobileNetV2, ModelName.MobileNetV3_Large, ModelName.MobileNetV3_Small, ModelName.InceptionV3]
  # model_name_list = [ModelName.DenseNet121, ModelName.EfficientNetB0, ModelName.ResNet50V1, ModelName.ResNet50V2, ModelName.NASNetMobile]
  model_name = ModelName.MobileNetV2
  model_name = ModelName.MobileNetV3_Small
  # model_name = ModelName.MobileNetV3_Small
  ConvertKerasModel(model_name=model_name)
  # for model_name in model_name_list:
  #   ConvertKerasModel(model_name=model_name)
  # CheckKerasModelOutput(model_name=model_name)

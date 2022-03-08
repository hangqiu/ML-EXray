import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd

# Output of the TFLite model.
tflite_model = "model/imagenet_mobilenet_v2_100_224_converted_tflite/mobilenet_v2_100_imagenet_quant_224.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
# Manually quantize the input from float to integer
scale, zero_point = input_details['quantization']
print(f"Input scale: {scale}, zero: {zero_point}")
# tflite_integer_input = tf_input / scale + zero_point
# tflite_integer_input = tflite_integer_input.astype(input_details['dtype'])
# interpreter.set_tensor(input_details['index'], tflite_integer_input)
# interpreter.invoke()
output_details = interpreter.get_output_details()[0]
# Manually dequantize the output from integer to float
scale, zero_point = output_details['quantization']
print(f"Output scale: {scale}, zero: {zero_point}")

tensor_details = interpreter.get_tensor_details()
for td in tensor_details:
    if td['name'] == 'mobilenetv2_1.00_224/global_average_pooling2d/Mean':
        print(td)
        break

tensor_details_pd = pd.DataFrame(tensor_details)
pd.set_option("display.max_rows", None)
print(tensor_details_pd)
# print(tensor_details)
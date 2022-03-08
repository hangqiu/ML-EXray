import os
import tensorflow as tf
from MLEXray.Model_Converter.Model_Converter import Model_Converter

model_dir = "model/SavedModels_TFHub/"

models = os.listdir(model_dir)

for saved_model_dir in models:
    names = saved_model_dir.split('_')
    count = len(names)
    version = names[-1]
    task = names[-2]
    model_name = '_'.join(names[1:-2])
    dataset = names[0]

    tflite_name = f"{model_dir}{saved_model_dir}/{model_name}_{dataset}_224"
    print(tflite_name)
    Model_Converter.SavedModelDir2Tflite(saved_model_dir=model_dir+saved_model_dir, model_name=tflite_name)
    tflite_name = f"{model_dir}{saved_model_dir}/{model_name}_{dataset}_quant_224"
    Model_Converter.SavedModelDir2Tflite(saved_model_dir=model_dir+saved_model_dir, model_name=tflite_name, quantize=True)
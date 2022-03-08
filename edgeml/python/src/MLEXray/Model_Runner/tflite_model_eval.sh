#!/bin/bash

# run in tensorflow root directory
# following the guide: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification#image-classification-evaluation-based-on-ilsvrc-2012-task

DEVKIT_DIR=/home/hang/tensorflow_datasets/imagenet2012/ILSVRC2012_devkit_t12/
VALIDATION_LABEL=/home/hang/tensorflow_datasets/imagenet2012/tflite_model_output_label.txt
TFLITE_MODEL=/home/hang/EdgeMLInsights/0_EdgeMLInsights/EdgeMLInsight/model/imagenet_mobilenet_v2_100_224_keras/mobilenet_v2_100_imagenet_224.tflite
MODEL_OUTPUT_LABEL=/home/hang/EdgeMLInsights/0_EdgeMLInsights/EdgeMLInsight/model/imagenet_label_1000.txt
IMAGES_DIR=/home/hang/tensorflow_datasets/downloads/manual/ILSVRC2012_img_val/

# generate groundtruth label
python3 tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification/generate_validation_labels.py \
  --ilsvrc_devkit_dir=${DEVKIT_DIR} \
  --validation_labels_output=${VALIDATION_LABEL}

# run evaluation on tflite model
bazel run -c opt   --   //tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval \
  --model_file=${TFLITE_MODEL}   \
  --ground_truth_images_path=${IMAGES_DIR}   \
  --ground_truth_labels=${VALIDATION_LABEL}   \
  --model_output_labels=${MODEL_OUTPUT_LABEL}   \
  --output_file_path=/tmp/accuracy_output.txt \
  --num_images=0 # ALL IMAGES

import json
import os
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from MLEXray.EdgeMLMonitor import EdgeMLMonitor
from MLEXray.Model_Runner.Image_Model_Runner import Image_Model_Runner
from MLEXray.Model_Runner.Model_Runner import Model_Runner
from MLEXray.Utils.params import ModelName, DatasetName
from MLEXray.Utils.tf_utils import KerasModel_preprocessing
import cv2
import tensorflow_hub as hub

# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

# Print Tensorflow version
print(tf.__version__)

# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class Tfhub_Detection_Model_Runner(Image_Model_Runner):
    embed_layer_name = None
    def __init__(self, model_name, eval=False):
        if model_name == ModelName.SSDMobileNetv2:
            self.detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
            self.input_size = 320
        elif model_name == ModelName.EfficientDet_D0:
            self.detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")
            self.input_size = 512
        elif model_name == ModelName.FasterRCNNResnet101v1:
            self.detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1")
            self.input_size = 640
        else:
            raise ValueError(f"Model name {self.model_name} not supported yet!")

        super().__init__(model_name)

    def invoke_model(self, input):
        result = self.detector(input)
        result['predictions'] = result['detection_scores']
        return result

    def convert_result_to_coco_format(self, img, img_meta, res):
        """

        :param img:
        :param res:
        :return: list of dict
        """
        all_res = []
        img_name = img.split('/')[-1].split('_')[-1].split('.')[0]
        img_name = int(img_name)

        # print(res['num_detections'].numpy())
        for i in range(int(res['num_detections'].numpy())):
            # print(i)
            # print(res['detection_classes'].shape)
            result = dict()
            result["image_id"] = img_name
            result["category_id"] = int(res['detection_classes'].numpy()[0][i])
            [ymin, xmin, ymax, xmax] = res['detection_boxes'].numpy()[0][i].tolist()
            # print(img_name)
            # print(img_meta)
            im_width = img_meta['size'][1]
            im_height = img_meta['size'][0]

            (xmin, xmax, ymin, ymax) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            [x, y, width, height] = [(xmax+xmin)/2, (ymax+ymin)/2, xmax-xmin, ymax-ymin]
            result["bbox"] = [xmin, ymin, width, height]
            result["score"] = float(res['detection_scores'].numpy()[0][i])
            # print(result)
            # raise
            all_res.append(result)

        return all_res

    def write_detection_result(self, imgs,img_metas, res, log_path):
        all_res = []
        for i in range(len(imgs)):
            res_list = self.convert_result_to_coco_format(imgs[i],img_metas[i], res[i])
            all_res.extend(res_list)

        f = open(log_path, "w")
        json.dump(all_res, f)
        f.close()

def run_detector(detector, path):
  img = load_img(path)

  converted_img  = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()

  result = {key:value.numpy()[0] for key,value in result.items()}

  print("Found %d objects." % len(result["detection_scores"]))
  print("Inference time: ", end_time-start_time)

  image_with_boxes = draw_boxes(
      img.numpy(), result["detection_boxes"],
      result["detection_classes"], result["detection_scores"])

  display_image(image_with_boxes)

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def display_image(image):
  plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)
  plt.savefig("tmp_tmp.png")
  plt.close()


def download_and_resize_image(url, new_width=256, new_height=256,
                              display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  response = urlopen(url)
  image_data = response.read()
  image_data = BytesIO(image_data)
  pil_image = Image.open(image_data)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image downloaded to %s." % filename)
  if display:
    display_image(pil_image)
  return filename


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(str(class_names[i]),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image


def coco_eval(res_file):
    annType = 'bbox'
    dataDir = 'data/0_data/coco2017'
    prefix = "instances"
    dataType = 'val2017'
    annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)

    cocoGt = COCO(annFile)
    cocoDt=cocoGt.loadRes(res_file)

    imgIds = sorted(cocoDt.getImgIds())
    # imgIds = imgIds[0:100]
    # imgId = imgIds[np.random.randint(100)]

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':
    # run data trace
    trace_name = DatasetName.COCO2017
    data_path = f"data/0_data/{trace_name}/images/val/"
    # trace_name = DatasetName.COCO2014_100
    # trace_name = DatasetName.COCO2017val_100
    # trace_name = DatasetName.COCO2017val_300

    # data_path = f"data/0_data/{trace_name}/images/"
    # model_name = ModelName.SSDMobileNetv2
    # model_name = ModelName.EfficientDet_D0
    model_name = ModelName.FasterRCNNResnet101v1

    res_file = f"data/trace_{trace_name}/{model_name}/1_cloud/{trace_name}/"
    # res_file = f"data/trace_{trace_name}/{model_name}/1_cloud_BGR/{trace_name}/"
    # res_file = f"data/trace_{trace_name}/{model_name}/1_cloud_Resize/{trace_name}/"
    # res_file = f"data/trace_{trace_name}/{model_name}/1_cloud_Normalization/{trace_name}/"
    if not os.path.exists(res_file):
        os.makedirs(res_file)
    res_file += "detection_result.json"
    # ref_res_file = "data/0_data/coco2017/results/instances_val2014_fakebbox100_results.json"

    runner = Tfhub_Detection_Model_Runner(model_name=model_name)

    imgs, img_metas, res = runner.run_image_data_folder(data_path, res_file, enableLog=False)
    runner.write_detection_result(imgs, img_metas, res, res_file)
    coco_eval(res_file)
    # coco_eval(ref_res_file)

    # single image test
    # run_detector(runner.detector, path="data/0_data/coco2017_100/images/000000000001.jpg")
    # run_detector(runner.detector, path="data/0_data/coco2014_100/images/COCO_val2014_000000000042.jpg")



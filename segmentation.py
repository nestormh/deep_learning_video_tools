# https://github.com/tensorflow/models/tree/master/research/deeplab
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf
import neuralgym.neuralgym as ng
from enum import Enum

import os

import labels

class modelName(Enum):
  deeplab_mnv3_large_cityscapes_trainfine = 'deeplab_mnv3_large_cityscapes_trainfine'
  mobilenetv2_coco_voctrainaug = 'mobilenetv2_coco_voctrainaug'
  mobilenetv2_coco_voctrainval = 'mobilenetv2_coco_voctrainval'
  xception_coco_voctrainaug = 'xception_coco_voctrainaug'
  xception_coco_voctrainval = 'xception_coco_voctrainval'

class DeepLabSegmentation(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, config):
    """Creates and loads pretrained deeplab model."""

    self.config = config

    self.graph = tf.Graph()

    tarball_path = self.download_model(self.config.model_name)

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        # graph_def = tf.GraphDef.FromString(file_handle.read())
        graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    # self.sess = tf.Session(graph=self.graph)
    self.sess = tf.compat.v1.Session(graph=self.graph)

  def download_model(self, modelName):
    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URLS = {
      'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
      'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
      'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
      'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
      'deeplab_mnv3_large_cityscapes_trainfine':
        'deeplab_mnv3_large_cityscapes_trainfine_2019_11_15.tar.gz',
    }
    _TARBALL_NAME = 'deeplab_model.tar.gz'

    model_dir = os.path.join(self.config.download_models_path, modelName)

    print(model_dir)

    download_path = os.path.join(model_dir, _TARBALL_NAME)
    if not os.path.exists(download_path):
      tf.io.gfile.makedirs(model_dir)
      print('downloading model, this might take a while...')
      urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[modelName], download_path)
      print('download completed! loading DeepLab model...')

    return download_path

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """

    # Convert from OpenCV to PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)

    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]

    # Convert from PIL to OpenCV
    resized_image = np.array(resized_image)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

    return resized_image, seg_map

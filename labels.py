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

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

# def generate_mask_from_labels(image, seg_map, labels):
#   """Visualizes input image, segmentation map and overlay view."""
#   plt.figure(figsize=(15, 5))
#   grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
#
#   plt.subplot(grid_spec[0])
#   plt.imshow(image)
#   plt.axis('off')
#   plt.title('input image')
#
#   plt.subplot(grid_spec[1])
#   seg_image = label_to_color_image(seg_map).astype(np.uint8)
#   plt.imshow(seg_image)
#   plt.axis('off')
#   plt.title('segmentation map')
#
#   plt.subplot(grid_spec[2])
#   plt.imshow(image)
#   plt.imshow(seg_image, alpha=0.7)
#   plt.axis('off')
#   plt.title('segmentation overlay')
#
#   unique_labels = np.unique(seg_map)
#   ax = plt.subplot(grid_spec[3])
#   plt.imshow(
#       FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
#   ax.yaxis.tick_right()
#   plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
#   plt.xticks([], [])
#   ax.tick_params(width=0.0)
#   plt.grid('off')
#   plt.show()

def mask_from_labels(seg_map, labels):
  colormap = np.zeros((256, 3), dtype=int)

  for label in labels:
    colormap[np.where(LABEL_NAMES == label), :] = 255

  seg_image = colormap[seg_map].astype(np.uint8)

  return seg_image



LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


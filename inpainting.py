import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from generative_inpainting.inpaint_model import InpaintCAModel
from timeit import default_timer as timer

class GenerativeInpainting:
    GRID_SIZE = 8

    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = config.checkpoint_path

        self.model = InpaintCAModel()

        self.sess_config = tf.compat.v1.ConfigProto()
        # TODO: Check
        self.sess_config.gpu_options.allow_growth = True
        self.sess_config.allow_soft_placement = True
        self.sess_config.log_device_placement = False

        self.sess = None
        self.output = None
        self.input_image_placeholder = None

        from pprint import pprint
        pprint(self.sess_config)

    def inpaint_image(self, image, mask):
        assert image.shape == mask.shape

        start_time = timer()

        h, w, _ = image.shape
        image_width = w//self.GRID_SIZE*self.GRID_SIZE
        image_height = h//self.GRID_SIZE*self.GRID_SIZE

        image = image[:image_height, :image_width, :]
        mask = mask[:image_height, :image_width, :]
        print(f'Shape of image: {image.shape}')

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        if not self.sess:
            self.sess = tf.compat.v1.Session(config=self.sess_config)

            self.input_image_placeholder = tf.compat.v1.placeholder(
                tf.float32, shape=(1, image_height, image_width * 2, 3))

            self.output = self.model.build_server_graph(self.config, self.input_image_placeholder)
            self.output = (self.output + 1.) * 127.5
            self.output = tf.reverse(self.output, [-1])
            self.output = tf.saturate_cast(self.output, tf.uint8)

            # load pretrained model
            vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)

            assign_ops = []
            for var in vars_list:
                vname = var.name

                # NOTE: Variables are not coincident with the provided model, so we will need to rename them by brute force
                if "upsample_conv" in vname:
                    vname = "inpaint_net/" + vname.split("/")[0][:-5] + "/" + vname
                else:
                    vname = "inpaint_net/" + vname

                from_name = vname

                var_value = tf.compat.v1.train.load_variable(self.checkpoint_dir, from_name)

                assign_ops.append(tf.compat.v1.assign(var, var_value))

            self.sess.run(assign_ops)

            print('Model loaded.')

        result = self.sess.run(self.output, feed_dict={self.input_image_placeholder: input_image})
        result = result[0][:, :, ::-1]

        print(f"Inpainting time: {timer() - start_time}")

        return result
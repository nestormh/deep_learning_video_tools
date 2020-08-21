import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym.neuralgym as ng

from generative_inpainting.inpaint_model import InpaintCAModel

class GenerativeInpainting:
    GRID_SIZE = 8

    def __init__(self, config_file, checkpoint_dir):
        self.config = ng.Config(config_file)
        self.checkpoint_dir = checkpoint_dir

        self.model = InpaintCAModel()

        self.sess_config = tf.compat.v1.ConfigProto()
        self.sess_config.gpu_options.allow_growth = True

    def inpaint_image(self, image, mask):
        assert image.shape == mask.shape

        h, w, _ = image.shape
        image = image[:h//self.GRID_SIZE*self.GRID_SIZE, :w//self.GRID_SIZE*self.GRID_SIZE, :]
        mask = mask[:h//self.GRID_SIZE*self.GRID_SIZE, :w//self.GRID_SIZE*self.GRID_SIZE, :]
        print(f'Shape of image: {image.shape}')

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        with tf.compat.v1.Session(config=self.sess_config) as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = self.model.build_server_graph(self.config, input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
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

            sess.run(assign_ops)

            print('Model loaded.')
            result = sess.run(output)

            result = result[0][:, :, ::-1]

            return result
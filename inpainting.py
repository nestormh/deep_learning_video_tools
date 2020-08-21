import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
import generative_inpainting.neuralgym.neuralgym as ng

from generative_inpainting.inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
# parser.add_argument('--image', default='generative_inpainting/examples/places2/case6_raw.png', type=str,
#                     help='The filename of image to be completed.')
# parser.add_argument('--mask', default='generative_inpainting/examples/places2/case6_mask.png', type=str,
#                     help='The filename of mask, value 255 indicates mask.')
# parser.add_argument('--output', default='/tmp/output.png', type=str,
#                     help='Where to write output.')

cases = [ "bike", "bike_square", "dogs", "dogs_squared", "personinroom", "wallstreet"]

case = cases[5]

parser.add_argument('--image', default=f'data/{case}_input.png', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default=f'data/{case}_mask.png', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default=f'data/{case}_inpainted.png', type=str,
                    help='Where to write output.')

parser.add_argument('--checkpoint_dir', default='models/inpainting/release_places2_256_deepfill_v2', type=str,
                    help='The directory of tensorflow checkpoint.')
# parser.add_argument('--checkpoint_dir', default='models/inpainting/release_celeba_hq_256_deepfill_v2', type=str,
#                     help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()

    model = InpaintCAModel()
    image = cv2.imread(args.image)
    mask = cv2.imread(args.mask)
    # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)

    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        sess.run(tf.compat.v1.global_variables_initializer())

        vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)

        print(vars_list)

        # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        #
        # latest_ckp = tf.train.latest_checkpoint(args.checkpoint_dir)
        # print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')
        #
        # exit(0)


        assign_ops = []
        for var in vars_list:
            vname = var.name

            # vname = vname.replace("kernel", "inpaint_net")
            if "upsample_conv" in vname:
                vname = "inpaint_net/" + vname.split("/")[0][:-5] + "/" + vname
            else:
                vname = "inpaint_net/" + vname

            from_name = vname


            print(f"vname {vname}")
            print(f"from_name {from_name}")
            # var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            var_value = tf.compat.v1.train.load_variable(args.checkpoint_dir, from_name)

            print(f"var_value {var_value}")
            assign_ops.append(tf.compat.v1.assign(var, var_value))

        print(f"assign_ops {assign_ops}")
        sess.run(assign_ops)

        print('Model loaded.')
        result = sess.run(output)

        print(result)
        cv2.imwrite(args.output, result[0][:, :, ::-1])

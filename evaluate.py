from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy

BATCH_SIZE = 4
DEVICE = '/gpu:0'

def ffwd(checkpoint_dir, device_t='/gpu:0', batch_size=1):

    img_shape = (200,200,3)

    g = tf.Graph()
    curr_num = 0
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        import cv2
        video_stream = cv2.VideoCapture(0)

        while(True):

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = video_stream.read()

            img = cv2.resize(frame, (200,200))
            cv2.imshow('frame', img)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            X = [img]

            _preds = sess.run(preds, feed_dict={img_placeholder:X})
            # save_img(path_out, _preds[j
            #print(_preds[0].shape)
            result = np.clip(_preds[0], 0, 255).astype(np.uint8)
            cv2.imshow('frame', result)

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--in-path', type=str,
                        dest='in_path',help='dir or file to transform',
                        metavar='IN_PATH', required=False)

    help_out = 'destination (dir or file) of transformed file or files'
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=False)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions',
                        help='allow different image dimensions')

    return parser

def main():
    parser = build_parser()
    opts = parser.parse_args()

    ffwd(opts.checkpoint_dir, device_t=opts.device)

if __name__ == '__main__':
    main()

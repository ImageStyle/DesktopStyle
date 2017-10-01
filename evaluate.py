from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import transform, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from argparse import ArgumentParser
from collections import defaultdict
import numpy as np

DEVICE = '/gpu:0'

def ffwd(checkpoint_dir, device_t='/gpu:0'):

    img_shape = (500,500,3)

    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (1,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_dir)

        import cv2
        video_stream = cv2.VideoCapture(0)

        while(True):

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = video_stream.read()

            img = cv2.resize(frame, (img_shape[0],img_shape[1]))
            cv2.imshow('frame', img)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            X = [img]

            _preds = sess.run(preds, feed_dict={img_placeholder:X})
            # save_img(path_out, _preds[j
            #print(_preds[0].shape)
            result = np.clip(_preds[0], 0, 255).astype(np.uint8)
            cv2.imshow('frame', cv2.resize(result, (img_shape[0],img_shape[1])))

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)
    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)
    return parser

def main():
    parser = build_parser()
    opts = parser.parse_args()

    ffwd(opts.checkpoint_dir, device_t=opts.device)

if __name__ == '__main__':
    main()

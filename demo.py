#!/usr/bin/env python2

import argparse
import tensorflow as tf
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

from tqdm import trange
from utils.config import Config
from model import ICNet, ICNet_BN

model_config = {
    'train': ICNet,
    'trainval': ICNet,
    'train_bn': ICNet_BN,
    'trainval_bn': ICNet_BN,
    'others': ICNet_BN
}

# Choose dataset here, but remember to use `script/downlaod_weight.py` first
dataset = 'cityscapes'
filter_scale = 1


class InferenceConfig(Config):
    def __init__(self, dataset, is_training, filter_scale):
        Config.__init__(self, dataset, is_training, filter_scale)

    # You can choose different model here, see "model_config" dictionary. If you choose "others",
    # it means that you use self-trained model, you need to change "filter_scale" to 2.
    model_type = 'trainval'

    # Set pre-trained weights here (You can download weight from Google Drive)
    model_weight = './model/cityscapes/icnet_cityscapes_trainval_90k.npy'

    # Define default input size here
    # INFER_SIZE = (1080, 1440, 3)  # height width
    INFER_SIZE = (1024, 2048, 3)


cfg = InferenceConfig(dataset, is_training=False, filter_scale=filter_scale)
cfg.display()

# Create graph here
model = model_config[cfg.model_type]
net = model(cfg=cfg, mode='inference')

# Create session & restore weight!
net.create_session()
net.restore(cfg.model_weight)

#------------------------------------------------------
# get ICNET output image and class
#------------------------------------------------------
for i in range(1, 13):
    image_path = '/home/nrsl/code/data/1440-1080-images/' + str(
        i) + 'image.png'
    im1 = cv2.imread(image_path)
    if im1.shape != cfg.INFER_SIZE:
        im1 = tf.gfile.FastGFile(image_path, 'rb').read()
        with tf.Session() as sess:
            img_after_decode = tf.image.decode_jpeg(im1)
            resized = tf.image.resize_images(img_after_decode, [1024, 2048],
                                             method=3)
            im1 = np.asarray(resized.eval(), dtype="uint8")

    results2, img_classes = net.predict(im1)
    print(img_classes)

    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    overlap_results2 = 0.5 * im1 + 0.5 * results2[0]

    vis_im2 = np.concatenate(
        [im1 / 255.0, results2[0] / 255.0, overlap_results2 / 255.0], axis=1)

    plt.figure(figsize=(20, 15))
    plt.imshow(vis_im2)
    plt.show()

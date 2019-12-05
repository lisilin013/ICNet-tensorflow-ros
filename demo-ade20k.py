#!/usr/bin/python3

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

class InferenceConfig(Config):
    def __init__(self, dataset, is_training, filter_scale):
        Config.__init__(self, dataset, is_training, filter_scale)

    # You can choose different model here, see "model_config" dictionary. If you choose "others",
    # it means that you use self-trained model, you need to change "filter_scale" to 2.
    model_type = 'trainval_bn'

    # Set pre-trained weights here (You can download weight from Google Drive)
    model_weight = './model/ade20k/model.ckpt-27150'

    # Define default input size here
    INFER_SIZE = (800, 800, 3) # height width


cfg = InferenceConfig('ade20k', is_training=False, filter_scale=1)

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
im2 = cv2.imread('./data/input/ladybug1.png')
results2, img_classes = net.predict(im2)
print(img_classes)

im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
overlap_results2 = 0.5 * im2 + 0.5 * results2[0]

vis_im2 = np.concatenate([im2 / 255.0, results2[0] / 255.0, overlap_results2 / 255.0], axis=1)

plt.figure(figsize=(20, 15))
plt.imshow(vis_im2)
plt.show()


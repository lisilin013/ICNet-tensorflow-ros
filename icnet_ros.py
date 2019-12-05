#!/usr/bin/env python
# coding:utf-8

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from utils.config import Config
from model import ICNet, ICNet_BN


class InferenceConfig(Config):
    def __init__(self, dataset, is_training, filter_scale):
        Config.__init__(self, dataset, is_training, filter_scale)

    # You can choose different model here, see "model_config" dictionary. If you choose "others",
    # it means that you use self-trained model, you need to change "filter_scale" to 2.
    model_type = 'trainval'

    # Set pre-trained weights here (You can download weight from Google Drive)
    model_weight = './model/cityscapes/icnet_cityscapes_trainval_90k.npy'

    # Define default input size here
    INFER_SIZE = (1080, 1440, 3)
    # INFER_SIZE = (1024, 2048, 3)


class ICNetRos:
    def __init__(self):
        # set config
        self.cfg = InferenceConfig(dataset='cityscapes',
                                   is_training=False,
                                   filter_scale=1)
        self.cfg.display()

        # Create graph here
        model_config = {
            'train': ICNet,
            'trainval': ICNet,
            'train_bn': ICNet_BN,
            'trainval_bn': ICNet_BN,
            'others': ICNet_BN
        }
        model = model_config[self.cfg.model_type]
        self.net = model(cfg=self.cfg, mode='inference')

        # Create session & restore weight!
        self.net.create_session()
        self.net.restore(self.cfg.model_weight)

    def get_infer_size(self):
        return self.cfg.INFER_SIZE

    def predict(self, img):
        '''
        @brief: get ICNET output image and class
        @return: semantic_img, classes_img, modify according to yourself
        '''
        start_t = time.time()
        semantic_img, classes_img, proba = self.net.predict(img)
        duration_t = time.time() - start_t

        # concatenate three images
        im2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlap_img = 0.5 * im2 + 0.5 * semantic_img[0]
        visual_img = np.concatenate([im2 / 255.0, semantic_img[0] / 255.0, overlap_img / 255.0], axis=1)
        rospy.loginfo('Predict: {:.4f} fps'.format(1 / duration_t))
        return visual_img, classes_img, proba


class ImagePredict:
    def __init__(self):
        self.image_pub = rospy.Publisher("visual_image", Image)
        self.bridge = CvBridge()
        self.icnet_ros = ICNetRos()
        rospy.loginfo('icnet ros init successfully!')
        # init finally
        self.image_sub = rospy.Subscriber("/camera_array/cam0/image_raw", Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)

            # INFER_SIZE = self.icnet_ros.get_infer_size() # (1024,2048,3)
            # if cv_image.shape != INFER_SIZE:
            #     cv_image = cv2.resize(cv_image, (INFER_SIZE[1], INFER_SIZE[0]), interpolation = cv2.INTER_AREA)
            
            visual_img, classes_img, proba = self.icnet_ros.predict(cv_image)
            cv2.namedWindow("visual_image", 0)
            cv2.imshow("visual_image", visual_img)
            cv2.resizeWindow("visual_image", 1440 * 3 / 2, 1080 / 2)
            rospy.loginfo(proba.shape)
            rospy.logwarn(proba)
            key_val = cv2.waitKey(1)

            # try:
            #     self.image_pub.publish(self.bridge.cv2_to_imgmsg(visual_img, "bgr8"))
            # except CvBridgeError as e:
            #     print(e)
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    rospy.init_node("icnet_ros_node")
    image_predict = ImagePredict()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

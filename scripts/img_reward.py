#!/usr/bin/env python

# img_reward.py: script to compute terminal reward from image
# Author: Nishanth Koganti
# Date: 2017/02/23

# import modules
import cv2
import rospy
import argparse
import numpy as np
import matplotlib.cm as cm
from sklearn import neighbors
from sensor_msgs.msg import Image
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError

rectX = 130
rectY = 134
imSize = 100

def computeImgReward(modelName,offsetX,offsetY):
    # load the model
    imgModel = joblib.load('%s.p' % (modelName))

    # get img message
    msg = rospy.wait_for_message("/kinect2/sd/image_depth_rect", Image, timeout=2)

    # convert msg to mat
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(msg, msg.encoding)
    rect = img[offsetX:offsetX+imSize,offsetY:offsetY+imSize]

    # process rect for model
    dat = ((rect*255.0)/4096.0).flatten()
    reward = imgModel.predict(np.atleast_2d(dat))

    return reward[0]

def main():
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', '-m', type=str, default='rewardModel',
                        help='Filename for terminal reward model')
    args = parser.parse_args()

    # initialize node
    rospy.init_node('ImgReward')

    # parsing the arguments
    modelName = args.modelname

    # call the function
    reward = computeImgReward(modelName,rectX,rectY)
    print 'Terminal Reward: %f' % (reward)

    return 0

if __name__ == "__main__":
    main()

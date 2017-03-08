#!/usr/bin/env python

# reward.py: utility functions to compute reward for reinforcement learning
# Author: Nishanth Koganti
# Date: 2017/03/02

# import modules
import cv2
import rospy
import numpy as np
import cPickle as pickle
from sensor_msgs.msg import Image
from sklearn.externals import joblib
from cv_bridge import CvBridge, CvBridgeError

def computeForceReward(fData, threshInd, fThresh, forceRate):
    # compute instant reward
    nSamples = fData['left'].shape[0]
    errLeft = np.maximum(fData['left']-fThresh['left'][:,0],0)
    errRight = np.maximum(fData['right']-fThresh['right'][:,0],0)
    fReward = np.exp(-forceRate*(errLeft+errRight))

    if threshInd < nSamples-1:
        fReward[threshInd+1:] = 0.0
    return fReward

def computeImgReward(imgData, termScale, modelName='rewardModel.p'):
    # estimate img reward
    imgModel = joblib.load(modelName)
    imgReward = termScale*imgModel.predict(imgData)
    return imgReward

def computeTermReward(offsetX, offsetY, termScale, modelName='rewardModel.p',
                      cutSize=250, imSize=100):
    # get img message
    msg = rospy.wait_for_message("/kinect2/sd/image_depth_rect", Image, timeout=2)
    msg2 = rospy.wait_for_message("/kinect2/qhd/image_color_rect", Image, timeout=2)

    # convert msg to mat
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(msg, msg.encoding)
    img2 = bridge.imgmsg_to_cv2(msg2, msg2.encoding)

    rect = img[offsetX:offsetX+cutSize,offsetY:offsetY+cutSize]
    frame = cv2.resize(rect, (imSize,imSize))

    color = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # process rect for model
    dat = np.atleast_2d(((frame*255.0)/4096.0).flatten()).astype(np.uint8)
    imgModel = joblib.load(modelName)
    reward = termScale*imgModel.predict(dat)
    return reward[0], color

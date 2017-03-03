#!/usr/bin/env python

# reward.py: utility functions to compute reward for reinforcement learning
# Author: Nishanth Koganti
# Date: 2017/03/02

# import modules
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def computeForceReward(fData, fThresh, threshInd=0):
    # compute instant reward
    errLeft = np.maximum(fData['left']-fThresh['left'][:,0],0)
    errRight = np.maximum(fData['right']-fThresh['right'][:,0],0)
    fReward = np.exp(-(errLeft+errRight))

    if threshInd > 0:
        fReward[threshInd:] = 0.0
    return fReward

def computeImgReward(imgData, imgModel, termScale=200.0):
    # estimate img reward
    imgReward = termScale*imgModel.predict(imgData)
    return imgReward

def computeTermReward(imgModel,offsetX,offsetY,imSize=100,termScale=200.0):
    # get img message
    msg = rospy.wait_for_message("/kinect2/sd/image_depth_rect", Image, timeout=2)

    # convert msg to mat
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(msg, msg.encoding)
    rect = img[offsetX:offsetX+imSize,offsetY:offsetY+imSize]

    # process rect for model
    dat = np.atleast_2d(((rect*255.0)/4096.0).flatten()).astype(np.uint8)
    reward = termScale*imgModel.predict(dat)
    return reward[0]

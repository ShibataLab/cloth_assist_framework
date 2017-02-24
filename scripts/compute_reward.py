#!/usr/bin/env python

# compute_reward.py: script to compute reward for current policy
# Author: Nishanth Koganti
# Date: 2017/02/17

# import modules
import cv2
import rospy
import argparse
import numpy as np
import cPickle as pickle
import matplotlib.cm as cm
from sklearn import neighbors
from sklearn.externals import joblib
from matplotlib import pyplot as plt

imSize = 100
nSamples = 200
termScale = 20.0
fInds = {'left':range(27,30),'right':range(33,36)}

def computeForceReward(fData, threshName, threshInd=0):
    # compute instant reward
    fThresh = pickle.load(open('%s.p' % (threshName), 'rb'))
    errLeft = np.maximum(fData['left']-fThresh['left'][:,0],0)
    errRight = np.maximum(fData['right']-fThresh['right'][:,0],0)
    fReward = np.exp(-(errLeft+errRight))

    if threshInd > 0:
        fReward[threshInd:] = 0.0
    return fReward

def computeImgReward(imgData, modelName):
    # estimate img reward
    imgModel = joblib.load('%s.p' % (modelName))
    imgReward = termScale*imgModel.predict(imgData)

    return imgReward

def main():
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', type=str, required=True,
                        help='Filename to be loaded')
    parser.add_argument('--modelname', '-m', type=str, default='rewardModel',
                        help='Filename for terminal reward model')
    parser.add_argument('--threshname', '-t', type=str, default='forceThresh',
                        help='Filename for force threshold values')
    args = parser.parse_args()

    # parsing the arguments
    fileName = args.filename
    modelName = args.modelname
    threshName = args.threshname

    # load the ee data
    eeData = np.genfromtxt('%sEE' % (fileName), delimiter=',', skip_header=1)
    indices = np.linspace(0, eeData.shape[0]-1, num=nSamples, endpoint=True,
                          dtype=np.int)
    eeData = eeData[indices,:]

    fData = {'left':np.linalg.norm(eeData[:,fInds['left']], axis=1),
             'right':np.linalg.norm(eeData[:,fInds['right']], axis=1)}

    # load and process image data
    imgData = np.zeros((0,imSize**2))
    capture = cv2.VideoCapture('%sDepth.avi' % (fileName))
    while True:
        ret,frame = capture.read()
        if ret == False:
            break
        sFrame = cv2.resize(frame, (imSize,imSize))
        gFrame = cv2.cvtColor(sFrame, cv2.COLOR_RGB2GRAY)
        imgData = np.vstack((imgData,gFrame.flatten()))
    capture.release()
    indices = np.linspace(0, imgData.shape[0]-1, num=nSamples/2, endpoint=True,
                          dtype=np.int)
    imgData = imgData[indices,:]

    # call the function
    fReward = computeForceReward(fData,threshName)
    imgReward = computeImgReward(imgData,modelName)
    imgReward = np.repeat(imgReward,2)

    reward = fReward.copy()
    reward[-1] = imgReward[-1]

    # plot the rewards for the trial
    plt.figure()
    plt.plot(np.arange(nSamples), fReward, '-b', label='Force')
    plt.plot(np.arange(nSamples), imgReward, '-g', label='Visual')
    plt.legend(loc=0)
    plt.show()

    # save the reward to file
    np.savetxt('%sReward' % fileName, reward, fmt='%.3f', delimiter=',')

    return 0

if __name__ == "__main__":
    main()

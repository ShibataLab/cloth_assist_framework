#!/usr/bin/env python

# process_video.py: script to process and save video as mat
# Author: Nishanth Koganti
# Date: 2017/02/17

# import modules
import cv2
import argparse
import numpy as np
import matplotlib.cm as cm
from sklearn import neighbors
from sklearn.externals import joblib
from matplotlib import pyplot as plt

alpha = 3.0
imSize = 100
nNeighbors = 5
sampleSize = 100

def generateModel(trialName, start, stop, modelName):
    # load the track file and get relevant frames
    track = np.genfromtxt(trialName,skip_header=1,delimiter=',')

    # load the video file and get data
    rawData = np.zeros((0,imSize**2))
    capture = cv2.VideoCapture('%sDepth.avi' % (trialName))

    # loop over the frames and save within time range
    while True:
        ret,frame = capture.read()
        if ret == False:
            break
        sFrame = cv2.resize(frame, (imSize,imSize))
        gFrame = cv2.cvtColor(sFrame, cv2.COLOR_RGB2GRAY)
        rawData = np.vstack((rawData,gFrame.flatten()))
    capture.release()

    # set stop time and get track indices
    stop = min(track[-1,1],stop)
    track = track[(track[:,1] >= start) & (track[:,1] <= stop)]

    # get training dataset for model
    startInd,stopInd = np.asarray(track[[0,-1],0],dtype=np.int)
    indices = np.linspace(startInd-1,stopInd-1,num=sampleSize,
                          endpoint=True,dtype=np.int)
    data = rawData[indices,:]

    # get training labels
    labels = np.exp(alpha*(np.linspace(0.0,1.0,num=sampleSize,endpoint=True)-1.0))

    # train knn regressor
    model = neighbors.KNeighborsRegressor(nNeighbors,weights='distance')
    model.fit(data,labels)

    # perform test prediction
    pred = model.predict(rawData)

    # verify results
    plt.figure()
    plt.plot(pred,'-b',linewidth=2,label='Terminal Reward')
    plt.legend(loc=0)
    plt.show()

    # save data matrix
    joblib.dump(model, modelName+'.p')

def main():
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', type=str, required=True,
                        help='Filename to be loaded')
    parser.add_argument('--modelname', '-m', type=str, default='rewardModel',
                        help='Filename for saving data')
    parser.add_argument('--starttime', '-t0', type=float, default=0.0,
                        help='Start time of trajectory (Default:0.0)')
    parser.add_argument('--stoptime', '-t1', type=float, default=100.0,
                        help='Stop time of trajectory (Default:100.0)')
    args = parser.parse_args()

    # parsing the arguments
    stopTime = args.stoptime
    trialName = args.filename
    modelName = args.modelname
    startTime = args.starttime

    # call the function
    generateModel(trialName,startTime,stopTime,modelName)

if __name__ == "__main__":
    main()

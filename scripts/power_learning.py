#!/usr/bin/env python

# power_learning.py: script to run power reinforcement learning algorithm
# Author: Nishanth Koganti
# Date: 2017/02/23

# import modules
import cv2
import time
import rospy
import argparse
import numpy as np
import matplotlib.cm as cm
from dmp_fit import dmpFit
from play import mapFile, rewindFile
from matplotlib import pyplot as plt
from term_reward import computeTermReward
from compute_reward import computeForceReward

nBFS = 50
nTrajs = 5
rectX = 130
rectY = 134
nIters = 20
nSamples = 200
modelName = 'rewardModel'
threshName = 'forceThresh'

def powerLearning(fileName):
    # load trajectory and keys
    data = np.genfromtxt(fileName, delimiter=',', names=True)

    keys = list(data.dtype.names)
    data = data.view(np.float).reshape(data.shape + (-1,))

    # initialize policy by fitting dmp
    dmp, initTraj, initParams = dmpFit(data, nBFS)

    # variables for training dmps
    nParams = initParams.size
    nDims = initTraj.shape[1]-1

    # length of episode
    T = 1.0
    dT = T/nSamples

    # variable for cumilative rewards
    cReturns = np.zeros(nIters+1)
    sReturns = np.zeros((nIters+1,2))

    # additional parameters for Q-value function
    Q = np.zeros((nSamples,nIters+1))
    basis = np.tile(dmp.gen_psi(dmp.cs.rollout()),(1,nDims))

    # store parameter values for all iterations
    params = np.zeros((nParams,nIters+1))

    # set the exploration variance for parameters
    std = 0.1*initParams.mean()*np.ones(nParams)
    variance = (0.1*initParams.mean())**2*np.ones((nParams,1))

    # initialize parameter values
    params[:,0] = initParams.flatten()

    # generate dmp trajectory with current parameters
    currentParam = params[:,0]

    # play the initial trajectory
    threshInd, fDat = mapFile(initTraj, keys)
    time.sleep(2)
    if threshInd > 0:
        rewindFile(initTraj, keys, threshInd)

    # get reward for initial trajectory
    reward = computeForceReward(fDat, threshName, threshInd)
    termReward = computeTermReward(modelName, rectX, rectY)
    reward[-1] += termReward

    # initialize Q-values for first iteration
    for n in range(nSamples):
        Q[:-n-1,0] += reward[n]
    Q[:,0] /= nSamples

    # loop over the iterations
    for i in range(nIters):
        # compute reward for the trial
        cReturns[i] = Q[0,i]

        # update the sorted returns table
        sReturns[0,:] = np.asarray([i,cReturns[i]])
        sReturns = sReturns[sReturns[:,1].argsort()]

        # update to policy parameters
        paramNom = np.zeros(nParams)
        paramDNom = 1e-10*np.ones(nParams)

        # loop over best iterations
        for j in range(np.min((i,nTrajs))):
            # get the cost
            ind = sReturns[-1-j,0]
            cost = sReturns[-1-j,1]

            # compute temporary basis function values
            tempW = ((basis**2)/(np.atleast_2d(np.sum(((basis**2)*(np.ones((nSamples,1))*variance.T)),axis=1)).T*np.ones((1,nParams)))).T
            tempExplore = (np.ones((nSamples,1))*np.atleast_2d(params[:,np.int(ind)] - currentParam)).T
            tempQ = np.ones((nParams,1))*np.atleast_2d(Q[:,j])

            # update policy parameters
            paramDNom += np.sum(tempW*tempQ, axis=1)
            paramNom += np.sum(tempW*tempExplore*tempQ, axis=1)

        # update policy parameters
        params[:,i+1] = currentParam + paramNom/paramDNom
        currentParam = params[:,i+1]

        # add exploration noise to next parameter set
        if i != nIters-1:
            params[:,i+1] = params[:,i+1] + std*np.random.randn(nParams)

        # generate rollout from parameters
        dmp.w = np.reshape(params[:,i+1],(nDims,nBFS))
        dmpTraj,_,_ = dmp.rollout()
        print dmpTraj.shape
        dmpTraj = np.hstack((np.atleast_2d(initTraj[:,0]).T,dmpTraj))
        # play trajectory and get reward
        threshInd, fDat = mapFile(dmpTraj, keys)
        time.sleep(2)
        if threshInd > 0:
            rewindFile(dmpTraj, keys, threshInd)

        reward = computeForceReward(fDat, threshName, threshInd)
        termReward = computeTermReward(modelName, rectX, rectY)
        reward[-1] += termReward

        # compute final reward
        for n in range(nSamples):
            Q[:-n-1,i+1] += reward[n]
        Q[:,i+1] /= nSamples

        print cReturns[i]

    cReturns[nIters] = Q[0,nIters]

    # plot the return over rollouts
    plt.figure()
    plt.plot(cReturns,'-b',linewidth=2)
    plt.ylabel('return')
    plt.xlabel('rollouts')
    plt.show()

def main():
    # initialize argument parser
    argFmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class = argFmt, description = main.__doc__)

    # add arguments to parser
    parser.add_argument('-f', '--filename', type = str, help = 'Output Joint Angle Filename')

    # parsing arguments
    args = parser.parse_args(rospy.myargv()[1:])

    # initialize node with a unique name
    rospy.init_node("PoWERLearning")

    powerLearning(args.filename)

if __name__ == "__main__":
    main()

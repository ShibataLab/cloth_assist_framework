#!/usr/bin/env python

# power_learning.py: script to run power reinforcement learning algorithm
# Author: Nishanth Koganti
# Date: 2017/02/23

# import modules
import cv2
import argparse
import numpy as np
import matplotlib.cm as cm
from sklearn import neighbors
from sklearn.externals import joblib
from matplotlib import pyplot as plt

nBFS = 50
nTrajs = 5
nIters = 50
nSamples = 200

def powerLearning():
    # initialize parameters
    params = dmpParam
    inputTraj = dmpTraj

    # variables for training dmps
    nDims = inputTraj.shape[1]
    nParams = dmpParams['input']['JA'].size

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
    std = 10*initParams.mean()*np.ones(nParams)
    variance = (10*initParams.mean())**2*np.ones((nParams,1))

    # initialize parameter values
    params[:,0] = initParams.flatten()

    # generate dmp trajectory with current parameters
    currentParam = params[:,0]
    dmp.w = np.reshape(currentParam,(nDims,nBFS))
    dmpTraj,_,_ = dmp.rollout()

    # initialize Q-values for first iteration
    for n in range(nSamples):
        Q[:-n-1,0] += np.exp(-np.sqrt(np.sum((dmpTraj[n,:]-outputTraj[n,:])**2)))
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

        # compute final reward
        for n in range(nSamples):
            Q[:-n-1,i+1] += np.exp(-np.sqrt(np.sum((dmpTraj[n,:]-outputTraj[n,:])**2)))
        Q[:,i+1] /= nSamples

    cReturns[nIters] = Q[0,nIters]
    Rewards['PoWER+Q'] = cReturns

    # plot the return over rollouts
    plt.figure()
    plt.plot(cReturns,'-b',linewidth=2)
    plt.ylabel('return')
    plt.xlabel('rollouts')

def main():


if __name__ == "__main__":
    main()

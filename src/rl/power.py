#!/usr/bin/env python

# power.py: power reinforcement learning agent with q-value approximation
# Author: Nishanth Koganti
# Date: 2017/03/03

# import modules
import time
import rospy
import argparse
import numpy as np
import cPickle as pickle
from matplotlib import pyplot as plt

class PowerAgent(object):
    def __init__(self, initParams, basis, nIters=20, nSamples = 400, nTrajs=5):
        # initialize parameters
        self.nIters = nIters
        self.nTrajs = nTrajs
        self.nSamples = nSamples
        self.nParams = initParams.size
        self.nBFs = initParams.shape[1]
        self.nDims = initParams.shape[0]

        # variable for cumilative rewards
        self.sReturns = np.zeros((self.nIters, 2))

        # additional parameters for Q-value function
        self.basis = np.tile(basis, (1, self.nDims))
        self.Q = np.zeros((self.nSamples, self.nIters))

        # store parameter values for all iterations
        self.params = np.zeros((self.nParams, self.nIters))

        # set the exploration variance for parameters
        self.std = 0.1*initParams.mean()*np.ones(self.nParams)
        self.variance = (0.1*initParams.mean())**2*np.ones((self.nParams, 1))

        # set the parameters for current iteration
        self.params[:,0] = initParams.flatten()
        self.currentParam = initParams.flatten()

        # set the iter variable
        self.iter = 0

    def update(self, reward):
        # initialize Q-values for first iteration
        for n in range(self.nSamples):
            self.Q[:-n, self.iter] += reward[n]
        self.Q[:, self.iter] /= self.nSamples

        # update the sorted returns table
        self.sReturns[0,:] = np.asarray([self.iter,self.Q[0,self.iter]])
        self.sReturns = self.sReturns[self.sReturns[:,1].argsort()]

        # update iter value
        self.iter += 1

        # return the cumilative reward
        return self.Q[0,self.iter-1]

    def sample(self):
        # update to policy parameters
        paramNom = np.zeros(self.nParams)
        paramDNom = 1e-10*np.ones(self.nParams)

        # loop over best iterations
        for j in range(np.min((self.iter,self.nTrajs))):
            # get the best trajectories
            ind = self.sReturns[-1-j,0]

            # compute temporary basis function values
            tempW = ((self.basis**2)/(np.atleast_2d(np.sum(((self.basis**2)*(np.ones((self.nSamples,1))*self.variance.T)),axis=1)).T*np.ones((1,self.nParams)))).T
            tempExplore = (np.ones((self.nSamples,1))*np.atleast_2d(self.params[:,np.int(ind)] - self.currentParam)).T
            tempQ = np.ones((self.nParams,1))*np.atleast_2d(self.Q[:,j])

            # update policy parameters
            paramDNom += np.sum(tempW*tempQ, axis=1)
            paramNom += np.sum(tempW*tempExplore*tempQ, axis=1)

        # update policy parameters
        self.params[:,self.iter] = self.currentParam + paramNom/paramDNom
        self.currentParam = self.params[:,self.iter]

        # add exploration noise to next parameter set
        if self.iter != self.nIters-1:
            self.params[:,self.iter] = self.params[:,self.iter] + self.std*np.random.randn(self.nParams)

        return self.params[:,self.iter].reshape((self.nDims,self.nBFs))

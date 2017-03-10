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
    def __init__(self, initParams, nIters=20, nSamples=400,
                 nTrajs=5, explParam=0.2, qMode=0, basis=None):
        # initialize parameters
        self.qMode = qMode
        self.nIters = nIters
        self.nTrajs = nTrajs
        self.nSamples = nSamples
        self.nParams = initParams.size
        self.nBFs = initParams.shape[1]
        self.nDims = initParams.shape[0]

        # variable for cumilative rewards
        self.sReturns = np.zeros((self.nIters, 2))

        # additional parameters for Q-value function
        if self.qMode:
            self.basis = np.tile(basis, (1, self.nDims))
            self.Q = np.zeros((self.nSamples, self.nIters))
        else:
            self.Returns = np.zeros(self.nIters)

        # store parameter values for all iterations
        self.params = np.zeros((self.nParams, self.nIters))

        # set the exploration variance for parameters
        self.std = explParam*initParams.mean()*np.ones(self.nParams)
        self.variance = (explParam*initParams.mean())**2*np.ones((self.nParams, 1))

        # set the parameters for current iteration
        self.params[:,0] = initParams.flatten()
        self.currentParam = initParams.flatten()

        # set the iter variable
        self.iter = 0

    def update(self, reward):
        if self.qMode:
            # initialize Q-values for first iteration
            self.Q[:,self.iter] = np.cumsum(reward[::-1])[::-1]

            # update the sorted returns table
            self.sReturns[0,:] = np.asarray([self.iter,self.Q[0,self.iter]])
            self.sReturns = self.sReturns[self.sReturns[:,1].argsort()]
        else:
            # compute the return for the trajectory
            self.Returns[self.iter] = np.sum(reward)/self.nSamples;

            # update the sorted returns table
            self.sReturns[0,:] = np.asarray([self.iter,self.Returns[self.iter]])
            self.sReturns = self.sReturns[self.sReturns[:,1].argsort()]

        # update iter value
        self.iter += 1

        # return the cumilative reward
        if self.qMode:
            return self.Q[0,self.iter-1]
        else:
            return self.Returns[self.iter-1]

    def sample(self):
        # update to policy parameters
        paramNom = np.zeros(self.nParams)
        if self.qMode:
            paramDNom = 1e-10*np.ones(self.nParams)
        else:
            paramDNom = 1e-10

        # loop over best iterations
        for j in range(np.min((self.iter,self.nTrajs))):
            # get the best trajectories
            ind = self.sReturns[-1-j,0]

            if self.qMode:
                # compute temporary basis function values
                tempW = ((self.basis**2)/(np.atleast_2d(np.sum(((self.basis**2)*(np.ones((self.nSamples,1))*self.variance.T)),axis=1)).T*np.ones((1,self.nParams)))).T
                tempExplore = (np.ones((self.nSamples,1))*np.atleast_2d(self.params[:,np.int(ind)] - self.currentParam)).T
                tempQ = np.ones((self.nParams,1))*np.atleast_2d(self.Q[:,j])

                # update policy parameters
                paramDNom += np.sum(tempW*tempQ, axis=1)
                paramNom += np.sum(tempW*tempExplore*tempQ, axis=1)
            else:
                # compute exploration w.r.t to current params
                tempExplore = self.params[:,np.int(ind)] - self.currentParam

                # update policy parameters
                paramDNom += self.Returns[np.int(ind)]
                paramNom += tempExplore*self.Returns[np.int(ind)]

        # update policy parameters
        self.params[:,self.iter] = self.currentParam + paramNom/paramDNom
        self.currentParam = self.params[:,self.iter]

        # update variance parameters
        if self.iter > 1:
            varNom = np.zeros(self.nParams)
            if self.qMode:
                varDNom = 1e-10*np.ones(self.nParams)
            else:
                varDNom = 1e-10

            # loop over best iterations
            for j in range(np.min((self.iter,self.nIters/2))):
                # get the best trajectories
                ind = self.sReturns[-1-j,0]

                if self.qMode:
                    # compute temporary basis function values
                    tempExplore = (np.ones((self.nSamples,1))*np.atleast_2d(self.params[:,np.int(ind)] - self.currentParam)).T
                    tempQ = np.ones((self.nParams,1))*np.atleast_2d(self.Q[:,j])

                    # update variance parameters
                    varDNom += np.sum(tempQ, axis=1)
                    varNom += np.sum(tempExplore**2*tempQ, axis=1)
                else:
                    # compute exploration w.r.t current params
                    tempExplore = self.params[:,np.int(ind)] - self.currentParam

                    # update policy parameters
                    varDNom += self.Returns[np.int(ind)]
                    varNom += tempExplore**2*self.Returns[np.int(ind)]

            # limit the variance that is produced
            varParams = np.minimum(np.maximum(varNom/varDNom,0.1*self.variance[:,0]), 10.0*self.variance[:,0])
        else:
            varParams = self.variance[:,0]

        # add exploration noise to next parameter set
        if self.iter != self.nIters-1:
            self.params[:,self.iter] = self.params[:,self.iter] + \
            np.sqrt(varParams)*np.random.randn(self.nParams)
        else:
            print 'Policy Evaluation!'
        return self.params[:,self.iter].reshape((self.nDims,self.nBFs))

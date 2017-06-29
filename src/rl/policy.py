#!/usr/bin/env python

# policy.py: utility functions for policy estimation using dmp
# Author: Nishanth Koganti
# Date: 2017/03/02
# Source: own code

# import modules
import numpy as np
from dmp.dmp_discrete import DMPs_discrete

armDims = range(1,8)
jointMap = np.atleast_2d([-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0])

class Policy(object):
    def __init__(self, data, nDims=2, nBFs=50):
        # set parameters to train dmp
        dt = 1.0/data.shape[0]

        # obtain active dimensions
        self.nBFs = nBFs
        self.nDims = nDims
        self.activeDims = np.sort(np.argsort(np.var(data[:,armDims],axis=0))[-self.nDims:])

        # setup and train the DMP
        self.dmp = DMPs_discrete(dmps=self.nDims, bfs=self.nBFs, dt=dt)
        self.dmp.imitate_path(y_des=np.transpose(data[:,self.activeDims+1]))
        self.params = self.dmp.w

        # generate a rollout from trained DMP
        dmpTraj,_,_ = self.dmp.rollout()
        armTraj = data[:,armDims].copy()
        armTraj[:,self.activeDims] = dmpTraj.copy()
        self.traj = np.hstack((np.atleast_2d(data[:,0]).T, armTraj, armTraj*jointMap))
        self.init = self.traj.copy()

    def update(self, params):
        self.dmp.w = params
        self.params = params

        dmpTraj,_,_ = self.dmp.rollout()
        armTraj = self.traj[:,armDims].copy()
        armTraj[:,self.activeDims] = dmpTraj.copy()
        self.traj = np.hstack((np.atleast_2d(data[:,0]).T, armTraj, armTraj*jointMap))

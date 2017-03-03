#!/usr/bin/env python

# policy.py: utility functions for policy estimation using dmp
# Author: Nishanth Koganti
# Date: 2017/03/02
# Source: own code

# import modules
import numpy as np
from dmp.dmp_discrete import DMPs_discrete

def dmpFit(data, nBFS=50):
    # set parameters to train dmp
    dt = 1.0/data.shape[0]
    nDims = data.shape[1]-1

    # setup and train the DMP
    dmp = DMPs_discrete(dmps=nDims, bfs=nBFS, dt=dt)
    dmp.imitate_path(y_des=np.transpose(data[:,1:]))

    # generate a rollout from trained DMP
    dmpParam = dmp.w
    dmpTraj,_,_ = dmp.rollout()
    dmpTraj = np.concatenate((np.transpose(np.atleast_2d(data[:,0])), dmpTraj), axis=1)

    return dmp, dmpTraj, dmpParam

def initPolicy(data, nBFS=50):
    # set parameters to train dmp
    dims=range(1,8)
    nDims = len(dims)
    dt = 1.0/data.shape[0]
    jointMap = np.atleast_2d([-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0])

    # setup and train the DMP
    dmp = DMPs_discrete(dmps=nDims, bfs=nBFS, dt=dt)
    dmp.imitate_path(y_des=np.transpose(data[:,dims]))

    # generate a rollout from trained DMP
    dmpParam = dmp.w
    dmpTraj,_,_ = dmp.rollout()
    dmpTraj = np.hstack((np.atleast_2d(data[:,0]).T, dmpTraj, dmpTraj*jointMap))

    return dmp, dmpTraj, dmpParam

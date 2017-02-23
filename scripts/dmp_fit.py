#!/usr/bin/env python

# dmp_fit.py: code to train dmp on raw trajectory
# Author: Nishanth Koganti
# Date: 2017/02/22
# Source: own code

# import modules
import argparse
import matplotlib
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from cloth_assist.plot_funcs import *
from cloth_assist.dmp_discrete import DMPs_discrete

def dmpFit(fileName, saveName, nBFS):
    # load the trajectory data
    data = np.genfromtxt(fileName, delimiter=',', names=True)

    header = ','.join(map(str, data.dtype.names))
    data = data.view(np.float).reshape(data.shape + (-1,))

    # set parameters to train dmp
    dt = 1.0/data.shape[0]
    nDims = data.shape[1]-1

    # setup and train the DMP
    dmp = DMPs_discrete(dmps=nDims, bfs=nBFS, dt=dt)
    dmp.imitate_path(y_des=np.transpose(data[:,1:]))

    # generate a rollout from trained DMP
    dmpParam = dmp.w
    dmpTraj,_,_ = dmp.rollout()
    dmpTraj = np.concatenate((np.transpose(np.atleast_2d(data[:,0])),dmpTraj),axis=1)

    # visualize trajectory
    plotTraj({'Raw':data,'DMP':dmpTraj}, colors={'Raw':'k','DMP':'b'})

    # save trajectory and parameters to files
    np.savetxt('%s' % (saveName), dmpTraj, fmt='%.3f', delimiter=',',
               header=header, comments='')
    np.savetxt('%sParam' % (saveName), dmpParam, fmt='%.3f', delimiter=',')

def main():
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', type=str, required=True,
                        help='Filename to be loaded')
    parser.add_argument('--savename', '-s', type=str, required=True,
                        help='Filename to be saved')
    parser.add_argument('--bfuncs', '-n', type=int, default=25,
                        help='Number of basis functions in DMP')

    args = parser.parse_args()

    # parsing the arguments
    nBFS = args.bfuncs
    fileName = args.filename
    saveName = args.savename

    # call the function
    dmpFit(fileName, saveName, nBFS)

if __name__ == "__main__":
    main()

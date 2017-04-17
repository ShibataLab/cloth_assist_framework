#!/usr/bin/env python

# power_learning.py: script to run power reinforcement learning algorithm
# Author: Nishanth Koganti
# Date: 2017/02/23

# import modules
import time
import rospy
import argparse
import numpy as np
import cPickle as pickle
from matplotlib import pyplot as plt

# import reinforcement learning modules
from rl.power import PowerAgent
from rl.policy import initPolicy
from rl.control import playFile, rewindFile
from rl.reward import computeForceReward, computeTermReward

# failure detection and reward parameters
rectX = 121
rectY = 107
forceRate = 10.0
termScale = 100.0
modelName = 'rewardModel.p'

# power implementation parameters
nTrajs = 5
nIters = 20
nSamples = 400
explParam = 0.5

# control parameters
forceThresh = 1.0

# policy parametrization parameters
nBFs = 50
jointMap = np.atleast_2d([-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0])

def moving_average(a, n=3) :
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:,:] = ret[n:,:] - ret[:-n,:]
    return np.pad(ret[n-1:,:]/n,((0,n-1),(0,0)),'edge')

def learn(fileName, forceName):
    # load initial trajectory
    data = np.genfromtxt(fileName, delimiter=',', names=True)

    keys = list(data.dtype.names)
    data = data.view(np.float).reshape(data.shape + (-1,))

    fThresh = pickle.load(open(forceName,'rb'))

    # initialize policy by fitting dmp
    dmp, initTraj, initParams = initPolicy(data, nBFs)
    basis = dmp.gen_psi(dmp.cs.rollout())

    # initialize power agent
    agent = PowerAgent(initParams, basis, nIters, nSamples, nTrajs, explParam)
    cReturns = np.zeros(nIters)

    # play the initial trajectory
    threshInd, fDat = playFile(initTraj, keys, 1, fThresh, forceThresh)
    time.sleep(2)

    # get reward for initial trajectory
    reward = computeForceReward(fDat, threshInd, fThresh, forceRate)
    termReward, termDat = computeTermReward(rectX, rectY, termScale, modelName)
    reward[threshInd] += termReward

    # rewind trajectory
    if threshInd > 0:
        rewindFile(initTraj, keys, threshInd)
        time.sleep(5)

    # update agent based on rewards observed
    cReturns[0] = agent.update(reward)

    # save the results for initialization
    results = {}
    results['force'] = fDat
    results['term'] = termDat
    results['traj'] = initTraj
    results['reward'] = reward
    results['qval'] = agent.Q[:,0]
    pickle.dump(results,open('Iter0.p','wb'))

    # loop over the iterations
    for i in range(nIters-1):
        # sample from the agent
        dmp.w = agent.sample()

        # generate rollout from parameters
        dmpTraj,_,_ = dmp.rollout()
        dmpTraj = np.hstack((np.atleast_2d(initTraj[:,0]).T, dmpTraj, dmpTraj*jointMap))

        # play trajectory and get reward
        threshInd, fDat = playFile(dmpTraj, keys, 1, fThresh, forceThresh)
        time.sleep(2)

        # compute reward obtained for trajectory
        reward = computeForceReward(fDat, threshInd, fThresh, forceRate)
        termReward, termDat = computeTermReward(rectX, rectY, termScale, modelName)
        reward[threshInd] += termReward

        # update the agent
        cReturns[i+1] = agent.update(reward)

        results = {}
        results['force'] = fDat
        results['term'] = termDat
        results['traj'] = dmpTraj
        results['reward'] = reward
        results['qval'] = agent.Q[:,i+1]
        pickle.dump(results,open('Iter%d.p' % (i+1),'wb'))

        # rewind trajectory if fail
        if threshInd > 0:
            rewindFile(initTraj, keys, threshInd)
            time.sleep(5)
        else:
            break

    # plot the return over rollouts
    plt.figure()
    plt.plot(cReturns[:i+1],'-b',linewidth=2)
    plt.ylabel('return')
    plt.xlabel('rollouts')
    plt.show()

def main():
    # initialize argument parser
    argFmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class = argFmt, description = main.__doc__)

    # add arguments to parser
    parser.add_argument('-t', '--trajname', type = str, help = 'Trajectory Filename')
    parser.add_argument('-f', '--forcename', type = str, help = 'Force Data Filename')

    # parsing arguments
    args = parser.parse_args(rospy.myargv()[1:])

    # initialize node with a unique name
    rospy.init_node("Learn")
    learn(args.trajname, args.forcename)

if __name__ == "__main__":
    main()

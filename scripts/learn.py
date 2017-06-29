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
from utility.plot_funcs import plotTraj

# import reinforcement learning modules
from rl.policy import Policy
from rl.power import PowerAgent
from rl.control import playFile, rewindFile
from rl.reward import computeForceReward, computeTermReward

# failure detection and reward parameters
rectX = 121
rectY = 107
forceRate = 10.0
termScale = 10.0
forceName = 'forceThresh.p'
modelName = 'rewardModel.p'

# power implementation parameters
qMode = 1
nTrajs = 5
nIters = 20
varMode = 0
nSamples = 400
explParam = 1.0

# control parameters
fixThresh = 0
forceThresh = 3.0

# policy parametrization parameters
nBFs = 50
nDims = 2

def learn(fileName, forceName):
    # load initial trajectory
    data = np.genfromtxt(fileName, delimiter=',', names=True)

    keys = list(data.dtype.names)
    data = data.view(np.float).reshape(data.shape + (-1,))

    if fixThresh:
        fThresh = {'left':forceThresh*np.ones((nSamples,1)),
                   'right':forceThresh*np.ones((nSamples,1))}
    else:
        fThresh = pickle.load(open(forceName,'rb'))

    # initialize policy by fitting dmp
    policy = Policy(data, nDims, nBFs)
    basis = policy.dmp.gen_psi(policy.dmp.cs.rollout())

    # initialize power agent
    agent = PowerAgent(initParams, basis, nIters, nSamples, nTrajs, explParam, varMode)
    cReturns = np.zeros(nIters)

    # play the initial trajectory
    threshInd, fDat = playFile(data, keys, 1, fThresh, forceThresh)
    time.sleep(2)

    # get reward for initial trajectory
    reward = computeForceReward(fDat, threshInd, fThresh, forceRate)
    termReward, termDat = computeTermReward(rectX, rectY, termScale, modelName)
    reward[threshInd] += termReward

    # rewind trajectory
    if threshInd < nSamples-1:
        rewindFile(initTraj, keys, threshInd)
        time.sleep(5)
    else:
        return 0

    # update agent based on rewards observed
    cReturns[0] = agent.update(reward)

    # save the results for initialization
    results = {}
    results['force'] = fDat
    results['term'] = termDat
    results['reward'] = reward
    results['traj'] = policy.traj
    if qMode:
        results['qval'] = agent.Q[:,0]
    else:
        results['return'] = agent.Returns[0]
    pickle.dump(results,open('Iter0.p','wb'))

    # loop over the iterations
    for i in range(nIters-1):
        # sample from the agent
        currentParams = agent.sample()
        policy.update(currentParams)

        # play trajectory and get reward
        print policy.traj
        threshInd, fDat = playFile(policy.traj, keys, 1, fThresh, forceThresh)
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
        results['reward'] = reward
        results['traj'] = policy.traj
        if qMode:
            results['qval'] = agent.Q[:,i]
        else:
            results['return'] = agent.Returns[i]
        pickle.dump(results,open('Iter%d.p' % (i+1),'wb'))

        # rewind trajectory if fail
        if threshInd < nSamples-1:
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

    return 0

def main():
    # initialize argument parser
    argFmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class = argFmt, description = main.__doc__)

    # add arguments to parser
    parser.add_argument('-t', '--trajname', type = str, help = 'Trajectory Filename')

    # parsing arguments
    args = parser.parse_args(rospy.myargv()[1:])

    # initialize node with a unique name
    rospy.init_node("Learn")
    learn(args.trajname)

if __name__ == "__main__":
    main()

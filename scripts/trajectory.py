#!/usr/bin/env python
"""program to process and plot baxter trajectory data."""

# trajectory_plot.py: python function to load and plot baxter trajectory data
# Author: Nishanth Koganti
# Date: 2017/02/23

# import modules
import argparse
import matplotlib
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

# matplotlib default settings
font = {'family': 'sans serif',
        'weight': 'bold',
        'size':   10}

matplotlib.rc('font', **font)

# constant offsets across functions
LEFT_ANGLE_OFFSET = 1
RIGHT_ANGLE_OFFSET = 8

LEFT_TORQUE_OFFSET = 15
RIGHT_TORQUE_OFFSET = 22

LEFT_EE_POS_OFFSET = 1
RIGHT_EE_POS_OFFSET = 8

LEFT_EE_FORCE_OFFSET = 27
RIGHT_EE_FORCE_OFFSET = 33

def plotJointAngles(angleData, jointIndex):
    """function to plot joint angle data."""
    # load all the joint data
    timeData = []
    jointData = []

    timeData.append(angleData[:, 0])
    timeData.append(angleData[:, 0])

    jointData.append(angleData[:, LEFT_ANGLE_OFFSET+jointIndex])
    jointData.append(angleData[:, RIGHT_ANGLE_OFFSET+jointIndex])

    # number of joints to plot
    nJoints = jointIndex.size
    arms = ['Left', 'Right']
    ylabels = ['Joint Angle (rad)', 'Joint Angle (rad)']

    # plot all the joint data
    for ind in range(2):
        plt.figure(figsize=(8, 2*nJoints))
        for i, jI in enumerate(jointIndex):
            plt.subplot(nJoints, 1, i+1)

            plt.plot(timeData[ind], jointData[ind][:, i], '-b', label='Smooth',
                     linewidth=2)

            plt.xlabel('Time (sec)', fontsize=12, fontweight='bold')
            plt.ylabel(ylabels[ind], fontsize=12, fontweight='bold')
            plt.title('%s Joint %d' % (arms[ind], jI+1), fontsize=15,
                      fontweight='bold')

            plt.tight_layout()

    # show all the plots
    plt.show()

def plotJoints(angleData, torqueData, jointIndex):
    """function to plot joint angle and torque data."""
    # load all the joint data
    timeData = []
    jointData = []

    timeData.append(angleData[:, 0])
    timeData.append(angleData[:, 0])
    timeData.append(torqueData[:, 0])
    timeData.append(torqueData[:, 0])

    jointData.append(angleData[:, LEFT_ANGLE_OFFSET+jointIndex])
    jointData.append(angleData[:, RIGHT_ANGLE_OFFSET+jointIndex])
    jointData.append(torqueData[:, LEFT_TORQUE_OFFSET+jointIndex])
    jointData.append(torqueData[:, RIGHT_TORQUE_OFFSET+jointIndex])

    # number of joints to plot
    nJoints = jointIndex.size
    arms = ['Left', 'Right', 'Left', 'Right']
    ylabels = ['Joint Angle (rad)', 'Joint Angle (rad)',
               'Joint Torque (Nm)', 'Joint Torque (Nm)']

    # plot all the joint data
    for ind in range(4):
        plt.figure(figsize=(8, 2*nJoints))
        for i, jI in enumerate(jointIndex):
            plt.subplot(nJoints, 1, i+1)

            plt.plot(timeData[ind], jointData[ind][:, i], '-b', label='Smooth',
                     linewidth=2)

            plt.xlabel('Time (sec)', fontsize=12, fontweight='bold')
            plt.ylabel(ylabels[ind], fontsize=12, fontweight='bold')
            plt.title('%s Joint %d' % (arms[ind], jI+1), fontsize=15,
                      fontweight='bold')

            plt.tight_layout()

    # show all the plots
    plt.show()

def plotEE(eeData):
    """function to plot end-effector data."""
    # load all the joint data
    endeffectData = []
    timeData = eeData[:, 0]

    # create list of indices (6 for force)
    plotIndex1 = np.arange(7)
    plotIndex2 = np.arange(6)

    endeffectData.append(eeData[:, LEFT_EE_POS_OFFSET+plotIndex1])
    endeffectData.append(eeData[:, RIGHT_EE_POS_OFFSET+plotIndex1])
    endeffectData.append(eeData[:, LEFT_EE_FORCE_OFFSET+plotIndex2])
    endeffectData.append(eeData[:, RIGHT_EE_FORCE_OFFSET+plotIndex2])

    # number of joints to plot
    arms = ['Left', 'Right', 'Left', 'Right']
    ylabels = ['Position (m)', 'Position (m)', 'Force (N)', 'Force (N)']
    titles = ['Pos X', 'Pos Y', 'Pos Z', 'Ang X', 'Ang Y', 'Ang Z', 'Ang W']
    nPlots = [plotIndex1.size, plotIndex1.size, plotIndex2.size,
              plotIndex2.size]

    # plot all the joint data
    for ind in range(4):
        plt.figure(figsize=(8, 2*nPlots[ind]))
        for i in range(nPlots[ind]):
            plt.subplot(nPlots[ind], 1, i+1)

            plt.plot(timeData, endeffectData[ind][:, i], '-b', label='Smooth',
                     linewidth=2)

            plt.xlabel('Time (sec)', fontsize=12, fontweight='bold')
            plt.ylabel(ylabels[ind], fontsize=12, fontweight='bold')
            plt.title(arms[ind] + ' ' + titles[i], fontsize=15,
                      fontweight='bold')

            plt.tight_layout()

    # show all the plots
    plt.show()

def processJA(fileName, savePath, plotFlag, startTime, stopTime, nSamples, jointIndex):
    """function to process joint angle data of trajectory."""
    # load the data files
    data = np.genfromtxt(fileName, delimiter=',', names=True)

    # get single header string
    header = ','.join(map(str, data.dtype.names))

    # convert structured array to numpy array
    data = data.view(np.float).reshape(data.shape + (-1,))

    # trim the files and update the time tracks
    if startTime != 0:
        startInd = (np.abs(data[:, 0] - startTime)).argmin()
        stopInd = (np.abs(data[:, 0] - stopTime)).argmin()
        data = data[startInd:stopInd, :]
        data[:, 0] = data[:, 0] - data[0, 0]

    if nSamples != 0:
        indices = np.linspace(0, data.shape[0]-1, num=nSamples, dtype=np.int)
        data = data[indices,:]

    if plotFlag:
        # plot the joint angles
        print 'Plotting the joint angles'
        plotJointAngles(data, jointIndex)

    if startTime != 0 or nSamples != 0:
        np.savetxt('%s' % (savePath), data, delimiter=',', fmt='%.4f',
                   header=header, comments='')

def moving_average(a, n=3) :
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:,:] = ret[n:,:] - ret[:-n,:]
    return np.pad(ret[n-1:,:]/n,((0,n-1),(0,0)),'edge')

def processAll(fileName, savePath, plotFlag, startTime, stopTime, nSamples, jointIndex):
    """function to process all types of trajectory data."""
    # load the data files
    eeData = np.genfromtxt('%sEE' % (fileName), delimiter=',', names=True)
    angleData = np.genfromtxt('%sJA' % (fileName), delimiter=',', names=True)
    torqueData = np.genfromtxt('%sJT' % (fileName), delimiter=',', names=True)

    # get single header string
    eeHeader = ','.join(map(str, eeData.dtype.names))
    angleHeader = ','.join(map(str, angleData.dtype.names))
    torqueHeader = ','.join(map(str, torqueData.dtype.names))

    # convert structured array to numpy array
    eeData = eeData.view(np.float).reshape(eeData.shape + (-1,))
    angleData = angleData.view(np.float).reshape(angleData.shape + (-1,))
    torqueData = torqueData.view(np.float).reshape(torqueData.shape + (-1,))

    # process end-effector orientation data
    indices = [range(4,8),range(11,15)]
    for inds in indices:
        quatData = eeData[:,inds].copy()
        prev = quatData[0,:]
        nSamp = quatData.shape[0]
        for i in range(1,nSamp):
            curr = quatData[i,:]
            if np.sum((curr-prev)**2) > np.sum((curr+prev)**2):
                quatData[i,:] = -quatData[i,:]
            prev = curr
        eeData[:,inds] = quatData

    # trim the files and update the time tracks
    if startTime != 0:
        eeStartInd = (np.abs(eeData[:, 0] - startTime)).argmin()
        angleStartInd = (np.abs(angleData[:, 0] - startTime)).argmin()
        torqueStartInd = (np.abs(torqueData[:, 0] - startTime)).argmin()

        eeStopInd = (np.abs(eeData[:, 0] - stopTime)).argmin()
        angleStopInd = (np.abs(angleData[:, 0] - stopTime)).argmin()
        torqueStopInd = (np.abs(torqueData[:, 0] - stopTime)).argmin()

        eeData = eeData[eeStartInd:eeStopInd, :]
        eeData[:, 0] = eeData[:, 0] - eeData[0, 0]

        angleData = angleData[angleStartInd:angleStopInd, :]
        angleData[:, 0] = angleData[:, 0] - angleData[0, 0]

        torqueData = torqueData[torqueStartInd:torqueStopInd, :]
        torqueData[:, 0] = torqueData[:, 0] - torqueData[0, 0]

    if nSamples != 0:
        indices = np.linspace(0, eeData.shape[0]-1, num=nSamples, dtype=np.int, endpoint=True)
        eeData = eeData[indices,:]
        angleData = angleData[indices,:]
        torqueData = torqueData[indices,:]

    if plotFlag:
        # plot the joint angles
        print 'Plotting the joint angles and torques'
        plotJoints(angleData, torqueData, jointIndex)

        # plot the joint velocities and torques
        print 'Plotting the end effector position, force'
        plotEE(eeData)

    if startTime != 0 or nSamples != 0:
        np.savetxt('%sEE' % (savePath), eeData, delimiter=',', fmt='%.4f',
                   header=eeHeader, comments='')
        np.savetxt('%sJA' % (savePath), angleData, delimiter=',', fmt='%.4f',
                   header=angleHeader, comments='')
        np.savetxt('%sJT' % (savePath), torqueData, delimiter=',', fmt='%.4f',
                   header=torqueHeader, comments='')

        # process the force data
        forceThresh = {'left': np.atleast_2d(np.linalg.norm(moving_average(eeData[:,27:30], n=20),axis=1)).T,
                       'right': np.atleast_2d(np.linalg.norm(moving_average(eeData[:,33:36], n=20),axis=1)).T}
        pickle.dump(forceThresh,open('%sThresh.p' % (savePath),'wb'))

def main():
    """main function of program with user interface and data preprocessing."""
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', type=str, required=True,
                        help='Enter filename to be loaded')
    parser.add_argument('--mode', '-m', type=int, default=0,
                        help='Enter processing mode (0:JA, 1:All),(Default:0)')
    parser.add_argument('--starttime', '-t0', type=float, default=0.0,
                        help='Enter start time of trajectory (Default:0.0)')
    parser.add_argument('--stoptime', '-t1', type=float, default=100.0,
                        help='Enter stop time of trajectory (Default:100.0)')
    parser.add_argument('--savepath', '-s', type=str, default='temp',
                        help='Enter path to save snipped files (Default:temp)')
    parser.add_argument('--plotflag', '-p', type=int, default=1,
                        help='Enter flag to plot trajectory (Default:True)')
    parser.add_argument('--numsamples', '-n', type=int, default=0,
                        help='Enter number of samples for down sampling (Default:0)')
    parser.add_argument('--joints', '-j', nargs='+', default=range(7),
                        help='Enter joints for plotting (Default:1:7)')
    args = parser.parse_args()

    # parsing the arguments
    processMode = args.mode
    fileName = args.filename
    savePath = args.savepath
    plotFlag = args.plotflag
    stopTime = args.stoptime
    nSamples = args.numsamples
    startTime = args.starttime
    jointIndex = np.asarray(args.joints)

    if processMode:
        processAll(fileName, savePath, plotFlag, startTime, stopTime, nSamples,
                   jointIndex)
    else:
        processJA(fileName, savePath, plotFlag, startTime, stopTime, nSamples,
                  jointIndex)

if __name__ == "__main__":
    main()

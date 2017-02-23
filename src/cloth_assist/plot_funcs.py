#!/usr/bin/env python

# plotFuncs.py: plot functions for data inspection
# Author: Nishanth Koganti
# Date: 2016/02/01

import sys
import GPy
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

################################################################################
# Functions to visualize trajectory data
################################################################################

def plotTraj(Dataset, plotType = 0, jointIndex = np.arange(7),
             labels = ['Train','Test'], colors=['b','r']):
    """function to plot multiple joint tracks."""
    timeData = {}
    leftData = {}
    rightData = {}

    LEFT_ANGLE_OFFSET = 1
    RIGHT_ANGLE_OFFSET = 8

    # loop over first plotNum files
    for key,data in Dataset.iteritems():
        timeData[key] = data[:, 0]
        leftData[key] = data[:, LEFT_ANGLE_OFFSET+jointIndex]
        rightData[key] = data[:, RIGHT_ANGLE_OFFSET+jointIndex]

    jointData = [leftData, rightData]

    # number of joints to plot
    xlabel = 'Time(sec)'
    arms = ['Left', 'Right']
    nJoints = jointIndex.size
    if plotType == 0:
        ylabels = 7*['Joint Angle (rad)']
    else:
        ylabels = 3*['Position (m)']+4*['Angle (rad)']

    # plot all the joint data
    for ind in range(2):
        fig = plt.figure(figsize=(10, 2*nJoints))
        for i, jI in enumerate(jointIndex):
            plt.subplot(nJoints, 1, i+1)

            # plot all the tracks
            for key in Dataset.keys():
                timeDat = timeData[key]
                nSamples = jointData[ind][key].shape[0]
                plt.plot(timeDat, jointData[ind][key][:, i], label=key,
                         color=colors[key], linewidth=2)

            plt.xlabel(xlabel, fontsize=12, fontweight='bold')
            plt.ylabel(ylabels[i], fontsize=12, fontweight='bold')

            if plotType == 0:
                plt.title('%s Joint %d' % (arms[ind], jI+1), fontsize=15,
                          fontweight='bold')
            else:
                plt.title('%s Pose %d' % (arms[ind], jI+1), fontsize=15,
                          fontweight='bold')

            # plot legend only for 1st sub plot
            if i == 0:
                plt.legend()

        # adjust subplots for legend
        fig.subplots_adjust(top=0.96, right=0.8)
        plt.tight_layout()

    # show all the plots
    plt.show()

def plotLatentTraj(Dataset, points = None, colors={'Train':'b','Test':'r'}):
    """function to plot multiple joint tracks."""
    timeData = {}
    latentData = {}

    # loop over first plotNum files
    for key,data in Dataset.iteritems():
        timeData[key] = data[:, 0]
        latentData[key] = data[:, 1:]

    # number of latent dims to plot
    xlabel = 'Time(sec)'
    nDim = latentData[key].shape[1]
    ylabels = nDim*['Latent Position']

    # plot all the latent data
    fig = plt.figure(figsize=(10, 2*nDim))
    for i in range(nDim):
        plt.subplot(nDim, 1, i+1)

        # plot all the tracks
        for n,key in enumerate(Dataset.keys()):
            timeDat = timeData[key]
            nSamples = latentData[key].shape[0]
            plt.plot(timeDat, latentData[key][:, i], label=key,
                     color=colors[key], linewidth=2)

        if points:
            plt.plot(points[i][:, 0], points[i][:, 1], 'ob', markersize=15,
                     label='viapoints')

        plt.xlabel(xlabel, fontsize=12, fontweight='bold')
        plt.ylabel(ylabels[i], fontsize=12, fontweight='bold')
        plt.title('Dim %d' % (i+1), fontsize=15, fontweight='bold')

    # adjust subplots for legend
    plt.tight_layout()

    # show all the plots
    plt.show()

################################################################################
# Functions to visualize model parameters and latent spaces
################################################################################

def plotScales(model, yThresh=0.05):
    # get ARD weight parameters
    scales = model.kern.input_sensitivity(summarize=False)
    scales =  scales/scales.max()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.arange(1,scales.shape[0]+1)
    ax.bar(x, height=scales, width=0.8, align='center',
           color='b', edgecolor='k', linewidth=1.3)
    ax.plot([0.4, scales.shape[0]+0.6], [yThresh, yThresh],
            '--', linewidth=3, color='r')

    # setting the bar plot parameters
    ax.set_xlim(.4, scales.shape[0]+.6)
    ax.set_title(model.name, fontsize=25)
    ax.set_ylabel('ARD Weight', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xticks(xrange(1,scales.shape[0]+1))
    ax.set_xlabel('Latent Dimensions', fontsize=20)

    return ax

def plotLatent(model, trainInput, testInput, mode=2, plotIndices = [0, 1]):
    sTest = 200
    sTrain = 150
    resolution = 50

    testMarker = 's'
    trainMarker = 'o'

    nTest = testInput.shape[0]
    nTrain = trainInput.shape[0]
    testLabels = [(1,0,0)]*nTest
    trainLabels = [(0,0,1)]*nTrain

    # get latent space plot parameters
    if mode == 0:
        model = model['model']
        testData = model.transform(testInput)
        trainData = model.transform(trainInput)
    else:
        qDim = model.X.mean.shape[1]
        testData = np.zeros((testInput.shape[0], qDim))
        trainData = np.zeros((trainInput.shape[0], qDim))

        for n in range(trainInput.shape[0]):
            # infer latent position
            xTrain, _ = model.infer_newX(np.atleast_2d(trainInput[n,:]),
                                         optimize=False)

            # update parameter
            if mode == 1:
                trainData[n,:] = xTrain
            else:
                trainData[n,:] = xTrain.mean
            sys.stdout.write('.')
        sys.stdout.write('\n')

        for n in range(testInput.shape[0]):
            # infer latent position
            xTest, _ = model.infer_newX(np.atleast_2d(testInput[n,:]),
                                        optimize=True)

            # update parameter
            if mode == 1:
                testData[n,:] = xTest
            else:
                testData[n,:] = xTest.mean
            sys.stdout.write('.')
        sys.stdout.write('\n')

    # variables for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    qDim = trainData.shape[1]
    if mode == 2:
        scales = model.kern.input_sensitivity(summarize=False)
        plotIndices = np.argsort(scales)[-2:]
    input1, input2 = plotIndices

    # compute plot limits
    xmin, ymin = trainData[:, [input1, input2]].min(0)
    xmax, ymax = trainData[:, [input1, input2]].max(0)
    x_r, y_r = xmax-xmin, ymax-ymin
    xmin -= .1*x_r
    xmax += .1*x_r
    ymin -= .1*y_r
    ymax += .1*y_r

    if mode > 0:
        # plot the variance for the model
        def plotFunction(x):
            Xtest_full = np.zeros((x.shape[0], qDim))
            Xtest_full[:, [input1, input2]] = x
            _, var = model.predict(np.atleast_2d(Xtest_full))
            var = var[:, :1]
            return -np.log(var)

        x, y = np.mgrid[xmin:xmax:1j*resolution, ymin:ymax:1j*resolution]
        gridData = np.hstack((x.flatten()[:, None], y.flatten()[:, None]))
        gridVariance = (plotFunction(gridData)).reshape((resolution, resolution))

        varianceHandle = plt.imshow(gridVariance.T, interpolation='bilinear',
                                    origin='lower', cmap=cm.gray,
                                    extent=(xmin, xmax, ymin, ymax))

    testHandle = ax.scatter(testData[:, input1], testData[:, input2],
                            marker=testMarker, s=sTest, c=testLabels,
                            linewidth=.2, edgecolor='k', alpha=1.)
    trainHandle = ax.scatter(trainData[:, input1], trainData[:, input2],
                             marker=trainMarker, s=sTrain, c=trainLabels,
                             linewidth=.2, edgecolor='k', alpha=1.)

    ax.grid(b=False)
    ax.set_aspect('auto')
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('Latent Dimension %i' % (input1+1), fontsize=25,
                  fontweight='bold')
    ax.set_ylabel('Latent Dimension %i' % (input2+1), fontsize=25,
                  fontweight='bold')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    properties = {'weight':'bold','size':25}
    plt.legend([trainHandle, testHandle], ['Train', 'Test'], prop=properties)

    fig.canvas.draw()
    fig.tight_layout()
    fig.canvas.draw()
    plt.show()

    return ax

################################################################################
# Functions to visualize results
################################################################################

# function to plot error bars
def plotErrorBars(mE, vE, xLabels, legend, colors, ylabel='NRMSE',
                  legendLoc=3, title='Comparison', ylimit=[0.,1.],
                  xlimit=[-0.1,2.1]):

    fontSize = 25
    N = mE.shape[1]

    widthFull = 0.8
    width = widthFull/N
    buffer = (1.0 - widthFull)/2.0

    ind = np.arange(mE.shape[0])
    fig, ax = plt.subplots()

    for i in range(mE.shape[1]):
        err = ax.bar(buffer+ind+i*width, mE[:,i], yerr=vE[:,i], width=width,
                     color=colors[i], ecolor='k')

    ax.set_ylim(ylimit)
    ax.set_xlim(xlimit)
    ax.set_xticks(ind + 0.5)
    ax.set_ylabel(ylabel, fontsize=fontSize, fontweight='bold')
    ax.set_xticklabels(xLabels, fontsize=fontSize-5, fontweight='bold')
    ax.legend(legend, loc=legendLoc, fontsize=fontSize,
              prop = {'size':fontSize-5, 'weight':'bold'})

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontSize-5)

    plt.tight_layout()
    plt.show()
    return ax

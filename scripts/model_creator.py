#!/usr/bin/env python

# model_creator.py: Python script to train bgplvm model on baxter trajectory data
# Author: Nishanth Koganti
# Date: 2017/04/17

# Source: GPy toolbox

# import the modules
import GPy
import argparse
import numpy as np
import cPickle as pickle
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from GPy.plotting.matplot_dep.controllers.imshow_controller import ImshowController

def plotScales(scales, options, yThresh=0.05):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.arange(1,scales.shape[0]+1)
    ax.bar(x, height=scales, width=0.8, align='center', color='b', edgecolor='k',
           linewidth=1.3)
    ax.plot([0.4, scales.shape[0]+0.6], [yThresh, yThresh], '--', linewidth=3,
            color='r')

    # setting the bar plot parameters
    ax.set_xlim(.4, scales.shape[0]+.6)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xticks(xrange(1,scales.shape[0]+1))
    ax.set_title(options['title'], fontsize=25)
    ax.set_ylabel(options['ylabel'], fontsize=20)
    ax.set_xlabel('Latent Dimensions', fontsize=20)

    return ax

def plotLatent(inX, title, model=None, plotIndices=None):
    s = 100
    marker = 'o'
    resolution = 50
    qDim = model.X.mean.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if plotIndices == None:
        scales = model.kern.input_sensitivity(summarize=False)
        plotIndices = np.argsort(scales)[-2:]
        print plotIndices
    input1, input2 = plotIndices

    xmin, ymin = inX[0][:, [input1, input2]].min(0)
    xmax, ymax = inX[0][:, [input1, input2]].max(0)
    x_r, y_r = xmax-xmin, ymax-ymin
    xmin -= .1*x_r
    xmax += .1*x_r
    ymin -= .1*y_r
    ymax += .1*y_r

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
                                origin='lower', cmap=cm.gray, extent=(xmin, xmax, ymin, ymax))

    trainH = ax.scatter(inX[0][:, input1], inX[0][:, input2], marker=marker, s=s,
                        c='b', linewidth=.2, edgecolor='k', alpha=1.)
    testH = ax.scatter(inX[1][:, input1], inX[1][:, input2], marker=marker, s=s,
                       c='r', linewidth=.2, edgecolor='k', alpha=0.9)

    ax.grid(b=False)
    ax.set_aspect('auto')
    ax.set_title(title, fontsize=25)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('Latent Dimension %i' % (input1+1), fontsize=20)
    ax.set_ylabel('Latent Dimension %i' % (input2+1), fontsize=20)

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    fig.canvas.draw()
    fig.tight_layout()
    fig.canvas.draw()

    return ax

def main():
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--fileName', '-f', type=str, required=True,
                        help='Enter filename to be loaded')
    parser.add_argument('--saveName', '-s', type=str, required=True,
                        help='Enter filename to save trained model')
    parser.add_argument('--init', '-i', type=str, default='pca',
                        help='Enter initialization method for BGPLVM')
    parser.add_argument('--sampleRate', '-r', type=int, default=1,
                        help='Sampling rate of training dataset')
    parser.add_argument('--numIters', '-n', type=int, default=100,
                        help='iterations for training model')
    parser.add_argument('--numInd', '-q', type=int, default=100,
                        help='number of inducing points for approximation')
    parser.add_argument('--numPlot', '-p', type=int, default=100,
                        help='number of points for latent space plot')

    args = parser.parse_args()

    # parsing the arguments
    init = args.init
    nInd = args.numInd
    nPlot = args.numPlot
    nIters = args.numIters
    sRate = args.sampleRate
    inputVar = args.fileName
    outputVar = args.saveName

    # load joint angle data with header
    data = np.genfromtxt('%s' % (inputVar), delimiter=',', names=True)
    header = ','.join(map(str, data.dtype.names))
    data = data.view(np.float).reshape(data.shape + (-1,))

    # create training data
    trainData = data[::sRate,1:]

    # model name
    expName = 'cloth_assist_model'

    # set the overall parameters for bgplvm
    qDim = 5
    nSamples = trainData.shape[0]

    # set the number of inducing inputs
    nInducing = nInd

    # get latent points and scales

    if init == 'pca':
        pca = PCA(n_components=qDim)
        pca.fit(trainData)

        xData = pca.transform(trainData)
        scales = pca.explained_variance_ratio_
        scales = scales/scales.max()
    elif init == 'rand':
        xData = np.random.normal(0, 1, (trainData.shape[0],qDim))
        scales = np.ones(qDim)

    # setting up the kernel
    kernel = GPy.kern.RBF(qDim, variance=1., lengthscale=1./scales,
                          ARD = True)

    # initialize BGPLVM model
    bgplvmModel = GPy.models.BayesianGPLVM(trainData, input_dim = qDim,
                                           num_inducing = nInducing,
                                           kernel = kernel, X = xData)

    # train the model
    SNR = 10
    var = bgplvmModel.Y.var()

    bgplvmModel.rbf.variance.fix(var)
    bgplvmModel.Gaussian_noise.variance.fix(var/SNR)

    initIters = nIters
    bgplvmModel.optimize(messages = True, max_iters = initIters)

    # training without constraints
    trainIters = 500
    bgplvmModel.unconstrain_fixed()
    bgplvmModel.optimize(messages = True, max_iters = trainIters)

    # plot results
    bgplvmX = [bgplvmModel.X.mean[:nPlot,:], np.zeros((1,qDim))]

    scalesBGPLVM = bgplvmModel.kern.input_sensitivity(summarize=False)
    scalesBGPLVM =  scalesBGPLVM/scalesBGPLVM.max()

    options = {'title':'Bayesian GPLVM','ylabel':'ARD Weight'}
    plotScales(scalesBGPLVM,options)
    plotLatent(bgplvmX, 'Bayesian GPLVM', model=bgplvmModel)
    plt.show()

    # save the model to file
    pickle.dump(bgplvmModel,open('%s.p' % (outputVar),'wb'))

if __name__ == "__main__":
    main()

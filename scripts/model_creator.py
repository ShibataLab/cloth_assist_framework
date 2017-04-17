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
    ax.bar(x, height=scales, width=0.8, align='center', color='b', edgecolor='k', linewidth=1.3)
    ax.plot([0.4, scales.shape[0]+0.6], [yThresh, yThresh], '--', linewidth=3, color='r')

    # setting the bar plot parameters
    ax.set_xlim(.4, scales.shape[0]+.6)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xticks(xrange(1,scales.shape[0]+1))
    ax.set_title(options['title'], fontsize=25)
    ax.set_ylabel(options['ylabel'], fontsize=20)
    ax.set_xlabel('Latent Dimensions', fontsize=20)

    return ax

def plotLatent(inX, title, model=None, which_indices=[0,1], plot_inducing=False, plot_variance=False, max_points=[800,300]):
    s = 100
    marker = 'o'
    resolution = 50

    fig = plt.figure()
    ax = fig.add_subplot(111)

    input1, input2 = which_indices

    if inX[0].shape[0] > max_points[0]:
        print("Warning".format(inX[0].shape))
        subsample = np.random.choice(inX[0].shape[0], size=max_points[0], replace=False)
        inX[0] = inX[0][subsample]

    if inX[1].shape[0] > max_points[1]:
        print("Warning".format(inX[1].shape))
        subsample = np.random.choice(inX[1].shape[0], size=max_points[1], replace=False)
        inX[1] = inX[1][subsample]

    xmin, ymin = inX[0][:, [input1, input2]].min(0)
    xmax, ymax = inX[0][:, [input1, input2]].max(0)
    x_r, y_r = xmax-xmin, ymax-ymin
    xmin -= .1*x_r
    xmax += .1*x_r
    ymin -= .1*y_r
    ymax += .1*y_r
    print xmin, xmax, ymin, ymax

    if plot_variance:
        def plot_function(x):
            Xtest_full = np.zeros((x.shape[0], X.shape[1]))
            Xtest_full[:, [input1, input2]] = x
            _, var = model.predict(np.atleast_2d(Xtest_full))
            var = var[:, :1]
            return np.log(var)

        X = inX[0]
        view = ImshowController(ax,plot_function,(xmin,ymin,xmax,ymax),resolution,aspect='auto',interpolation='bilinear',cmap=cm.binary)

    trainH = ax.scatter(inX[0][:, input1], inX[0][:, input2], marker=marker, s=s, c='b', linewidth=.2, edgecolor='k', alpha=1.)
    testH = ax.scatter(inX[1][:, input1], inX[1][:, input2], marker=marker, s=s, c='r', linewidth=.2, edgecolor='k', alpha=0.9)

    ax.grid(b=False)
    ax.set_aspect('auto')
    ax.set_title(title, fontsize=25)
    ax.tick_params(axis='both', labelsize=20)
    #ax.legend([trainH,testH],['Train','Test'],loc=1,fontsize=20)
    ax.set_xlabel('Latent Dimension %i' % (input1+1), fontsize=20)
    ax.set_ylabel('Latent Dimension %i' % (input2+1), fontsize=20)

    if plot_inducing:
        Z = model.Z
        ax.scatter(Z[:, input1], Z[:, input2], c='w', s=25, marker="^", edgecolor='k', linewidth=.3, alpha=.6)

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    fig.canvas.draw()
    fig.tight_layout()
    fig.canvas.draw()

    return ax

def main():
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Enter filename to be loaded')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Enter filename to save trained model')
    args = parser.parse_args()

    # parsing the arguments
    inputVar = args.input
    outputVar = args.output

    # load joint angle data with header
    data = np.genfromtxt('%s' % (inputVar), delimiter=',', names=True)
    header = ','.join(map(str, data.dtype.names))
    data = data.view(np.float).reshape(data.shape + (-1,))

    # create training data
    trainData = data[:,1:]

    # model name
    expName = 'cloth_assist_model'

    # set the overall parameters for bgplvm
    qDim = 5
    nSamples = data.shape[0]

    # set the number of inducing inputs
    nInducing = 50

    pca = PCA(n_components=qDim)
    pca.fit(trainData)

    # get latent points and scales
    xData = pca.transform(trainData)
    scales = pca.explained_variance_ratio_
    scales = scales/scales.max()

    # setting up the kernel
    kernel = GPy.kern.RBF(qDim, variance=1., lengthscale=1./scales,
                          ARD = True)

    # initialize BGPLVM model
    bgplvmModel = GPy.models.BayesianGPLVM(trainData, input_dim = qDim,
                                           num_inducing = nInducing,
                                           kernel = kernel, X = xData)

    # train the model
    SNR = 100
    var = bgplvmModel.Y.var()

    bgplvmModel.rbf.variance.fix(var)
    bgplvmModel.Gaussian_noise.variance.fix(var/SNR)

    initVardistIters = 500
    bgplvmModel.optimize(messages = True, max_iters = initVardistIters)

    # training without constraints
    trainIters = 500
    bgplvmModel.unconstrain_fixed()
    bgplvmModel.optimize(messages = True, max_iters = trainIters)

    # plot results
    bgplvmX = [xData,np.zeros((1,qDim))]

    scalesBGPLVM = bgplvmModel.kern.input_sensitivity(summarize=False)
    scalesBGPLVM =  scalesBGPLVM/scalesBGPLVM.max()

    options = {'title':'Bayesian GPLVM','ylabel':'ARD Weight'}
    plotScales(scalesBGPLVM,options)
    plotLatent(bgplvmX, 'Bayesian GPLVM', model=bgplvmModel, plot_variance=False, max_points=[1800,400])
    plt.show()

    # save the model to file
    pickle.dump(bgplvmModel,open('%s.p' % (outputVar),'wb'))

if __name__ == "__main__":
    main()

#!/usr/bin/python

# model_animator.py: python script to load bgplvm model and construct joint angles
# Author: Nishanth Koganti
# Date: 2017/04/17

import os
import GPy
import rospy
import argparse
import numpy as np
import cPickle as pickle
from matplotlib import cm
import matplotlib.pyplot as plt
from sensor_msgs.msg import JointState
from GPy.plotting.matplot_dep.controllers.imshow_controller import ImshowController

joint_names = ['head_pan', 'l_gripper_l_finger_joint', 'l_gripper_r_finger_joint',
               'r_gripper_l_finger_joint', 'r_gripper_r_finger_joint', 'left_s0',
               'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2',
               'right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1',
               'right_w2']

RED = 90
GREEN = 50

class ModelAnimator(object):
    def __init__(self,fname):
        # load bgplvm model from pickle file
        self.model = pickle.load(open(fname,'rb'))

        # parameters for doing latent function inference
        self.qDim = self.model.X.mean.shape[1]

        # create a joint_states publisher
        self.statePub = rospy.Publisher('joint_states', JointState, latch=True, queue_size=3)

        # variables for generating motion
        self.xMove = 0.0
        self.yMove = 0.0
        self.startMotion = False

        # visualize the bgplvm latent space
        self.maxPoints = 100
        self.plotVariance = True
        scales = self.model.kern.input_sensitivity(summarize=False)
        self.plotIndices = np.argsort(scales)[-2:]
        self.title = 'Latent Space Visualization'

        # variables for real-time plotting
        self.testHandle = None
        self.pointerColor = GREEN
        self.pointerHandle = None
        self.testData = np.empty((0,2))

        self.ax = self.plotLatent()
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        self.textHandle = plt.text(xmax/2, ymin+0.3, 'Play Mode: OFF',
                                   bbox={'facecolor':'green', 'alpha':0.5, 'pad':10})

        # connect the cursor class
        plt.connect('button_press_event',self.mouseClick)
        plt.connect('motion_notify_event', self.mouseMove)
        plt.show()

        raw_input('Press enter to finish')

        return

    def plotLatent(self):
        s = 100
        marker = 'o'
        resolution = 50

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # get latent space plot parameters
        latentData = self.model.X.mean
        input1, input2 = self.plotIndices

        # subsample latent points for easier visualization
        if latentData.shape[0] > self.maxPoints:
            latentData = latentData[:self.maxPoints,:]

        # compute plot limits
        xmin, ymin = latentData[:, [input1, input2]].min(0)
        xmax, ymax = latentData[:, [input1, input2]].max(0)
        x_r, y_r = xmax-xmin, ymax-ymin
        xmin -= .1*x_r
        xmax += .1*x_r
        ymin -= .1*y_r
        ymax += .1*y_r

        # plot the variance for the model
        if self.plotVariance:
            def plotFunction(x):
                Xtest_full = np.zeros((x.shape[0], self.qDim))
                Xtest_full[:, [input1, input2]] = x
                _, var = self.model.predict(np.atleast_2d(Xtest_full))
                var = var[:, :1]
                return -np.log(var)

            x, y = np.mgrid[xmin:xmax:1j*resolution, ymin:ymax:1j*resolution]
            gridData = np.hstack((x.flatten()[:, None], y.flatten()[:, None]))
            gridVariance = (plotFunction(gridData)).reshape((resolution, resolution))

            varianceHandle = plt.imshow(gridVariance.T, interpolation='bilinear',
                                        origin='lower', cmap=cm.gray,
                                        extent=(xmin, xmax, ymin, ymax))

        dataHandle = ax.scatter(latentData[:, input1], latentData[:, input2],
                                marker=marker, s=s, c='b', linewidth=.2,
                                edgecolor='k', alpha=1.)

        self.testHandle = ax.scatter(self.testData[:, 0], self.testData[:, 1],
                                     marker=marker, s=s, c='r', linewidth=.2,
                                     edgecolor='k', alpha=0.5)

        ax.grid(b=False)
        ax.set_aspect('auto')
        ax.set_title(self.title, fontsize=25)
        ax.tick_params(axis='both', labelsize=20)
        # ax.legend([trainH,testH],['Train','Test'],loc=1,fontsize=20)
        ax.set_xlabel('Latent Dimension %i' % (input1+1), fontsize=20)
        ax.set_ylabel('Latent Dimension %i' % (input2+1), fontsize=20)

        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))

        fig.canvas.draw()
        fig.tight_layout()
        fig.canvas.draw()

        return ax

    def mouseMove(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        if np.sqrt((self.xMove-x)**2+(self.yMove-y)**2) > 0.05:
            # update class variables
            self.xMove = x
            self.yMove = y

            # reconstruct joint angle data
            latent_value = (self.qDim)*[0.0]
            latent_value[self.plotIndices[0]] = self.xMove
            latent_value[self.plotIndices[1]] = self.yMove
            joint_angles = self.model.predict(np.atleast_2d(latent_value))
            joint_angles = [0]*5 + joint_angles[0][0,:].tolist()

            # publishing the joint state
            js = JointState(name=joint_names, position=joint_angles)
            js.header.stamp = rospy.Time.now()
            self.statePub.publish(js)

            if self.pointerHandle:
                self.pointerHandle.set_offsets(np.atleast_2d([self.xMove,self.yMove]))
                self.pointerHandle.set_array(np.array([self.pointerColor]))
            else:
                self.pointerHandle = self.ax.scatter(self.xMove, self.yMove, marker='s', s=300, c=self.pointerColor, linewidth=.2,
                                                     edgecolor='k', alpha=1., vmin=0, vmax=100)

            #if self.startMotion:

            plt.draw()
            return

    def mouseClick(self, event):
        if not event.inaxes:
            return

        self.startMotion = ~self.startMotion
        if self.startMotion:
            self.pointerColor = RED
            self.textHandle.set_text("Play Mode: ON")
            self.textHandle.set_bbox({'facecolor':'red', 'alpha':0.5, 'pad':10})
            plt.draw()
        else:
            self.pointerColor = GREEN
            self.testData = np.empty((0,2))
            self.textHandle.set_text("Play Mode: OFF")
            self.textHandle.set_bbox({'facecolor':'green', 'alpha':0.5, 'pad':10})
            plt.draw()
        return

def main():
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, required=True,
                        help='Enter modelname to be loaded')
    args, unknown = parser.parse_known_args()

    rospy.init_node('model_animator', log_level=rospy.INFO)

    try:
        animator = ModelAnimator(args.file)
    except rospy.ROSInterruptException:
        pass
    rospy.spin()

if __name__ == '__main__':
    main()

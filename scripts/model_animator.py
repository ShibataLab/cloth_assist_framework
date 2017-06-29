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

class ModelAnimator(object):
    def __init__(self,fname):
        # load bgplvm model from pickle file
        self.model = pickle.load(open(fname,'rb'))

        # parameters for doing latent function inference
        self.qDim = self.model.X.mean.shape[1]

        # create a joint_states publisher
        self.state_pub = rospy.Publisher('joint_states', JointState, latch=True, queue_size=3)

        # visualize the latent space of trained model
        self.x = 0.0
        self.y = 0.0
        fig,ax = plt.subplots()
        self.model.plot_latent(ax=ax)

        # connect the cursor class
        plt.connect('motion_notify_event', self.mouse_move)
        plt.show()

        raw_input('Press enter to finish')

        return

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        if np.sqrt((self.x-x)**2+(self.y-y)**2) > 0.05:
            # update class variables
            self.x = x
            self.y = y

            # reconstruct joint angle data
            latent_value = [self.x,self.y] + (self.qDim-2)*[0.0]
            joint_angles = self.model.predict(np.atleast_2d(latent_value))
            joint_angles = [0]*5 + joint_angles[0][0,:].tolist()

            # publishing the joint state
            js = JointState(name=joint_names, position=joint_angles)
            js.header.stamp = rospy.Time.now()
            self.state_pub.publish(js)
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

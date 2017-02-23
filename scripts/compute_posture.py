#!/usr/bin/env python

# compute_posture.py: code to compute posture of mannequin
# Author: Nishanth Koganti
# Date: 2016/06/22
# Source: own code

# import modules
import tf
import yaml
import rospy
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# matplotlib settings
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

axes = {'titlesize'  : '15',
        'labelsize'  : '12',
        'titleweight': 'bold',
        'labelweight': 'bold'}

matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)

HEAD = 0
NECK = 5
BODY = 6
LEFTWRIST = 1
RIGHTWRIST = 3
BODYLENGTH = 0.5
LEFTSHOULDER = 2
RIGHTSHOULDER = 4

def main():
    '''Program to compute posture of mannequin and plot results'''

    # initialize ros node
    rospy.init_node('compute_posture')

    # get arguments
    resultsFile = rospy.get_param('~results_file')
    postureFile = rospy.get_param('~posture_file')
    calibrationFile = rospy.get_param('~calibration_file')

    # load transformation matrix from yaml file
    with open(calibrationFile, 'r') as f:
        params = yaml.load(f)

    # get rotation quaternion and translation vector
    trans = params['trans']
    rot = params['rot_euler']

    # construct transformation matrix
    transMatrix = tf.transformations.compose_matrix(translate = trans, angles = rot)

    # load posture points
    posturePoints = np.loadtxt(postureFile, delimiter = ',')
    posturePoints = np.hstack((posturePoints, np.ones((posturePoints.shape[0], 1))))

    # apply transformation matrix to points
    posturePoints = np.dot(transMatrix, posturePoints.T)
    posturePoints = posturePoints[:-1,:].T

    # compute neck and body points
    neck = (posturePoints[LEFTSHOULDER,:] + posturePoints[RIGHTSHOULDER,:])/2
    body = neck.copy()
    body[2] = body[2]-BODYLENGTH

    # append neck and body points
    posturePoints = np.vstack((posturePoints,neck,body))

    # compute posture centroid
    postureMean = posturePoints.mean(axis=0)

    # compute vectors
    headNeck = posturePoints[HEAD,:] - posturePoints[NECK,:]
    headNeck = headNeck/np.linalg.norm(headNeck)
    neckBody = posturePoints[NECK,:] - posturePoints[BODY,:]
    neckBody = neckBody/np.linalg.norm(neckBody)
    leftArm = posturePoints[LEFTWRIST,:] - posturePoints[LEFTSHOULDER,:]
    leftArm = leftArm/np.linalg.norm(leftArm)
    rightArm = posturePoints[RIGHTWRIST,:] - posturePoints[RIGHTSHOULDER,:]
    rightArm = rightArm/np.linalg.norm(rightArm)

    # compute head, left arm and right arm angles
    leftAngle = np.arccos(np.dot(leftArm,neckBody))*180/np.pi
    headAngle = np.arccos(np.dot(headNeck,neckBody))*180/np.pi
    rightAngle = np.arccos(np.dot(rightArm,neckBody))*180/np.pi

    # plot mannequin posture
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the data
    Axes3D.scatter(ax, posturePoints[:,0], posturePoints[:,1], posturePoints[:,2],
                   s = 50, c = 'r')
    Axes3D.plot(ax, posturePoints[[HEAD,NECK,BODY],0], posturePoints[[HEAD,NECK,BODY],1],
                posturePoints[[HEAD,NECK,BODY],2], linewidth = 3, c = 'b')
    Axes3D.plot(ax, posturePoints[[LEFTWRIST,LEFTSHOULDER,RIGHTSHOULDER,RIGHTWRIST], 0],
                posturePoints[[LEFTWRIST,LEFTSHOULDER,RIGHTSHOULDER,RIGHTWRIST], 1],
                posturePoints[[LEFTWRIST,LEFTSHOULDER,RIGHTSHOULDER,RIGHTWRIST], 2],
                linewidth = 3, c = 'b')

    # add labels and title
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.title('Mannequin Posture (%.1f,%.1f,%.1f)' % (headAngle,leftAngle,rightAngle))
    plt.show()

    # save results to yaml file
    results = dict(
        Location = postureMean.tolist(),
        Angles = dict(
            Head = float(headAngle),
            LeftArm = float(leftAngle),
            RightArm = float(rightAngle)
        ),
        Posture = dict(
            HEAD = posturePoints[HEAD,:].tolist(),
            NECK = posturePoints[NECK,:].tolist(),
            BODY = posturePoints[BODY,:].tolist(),
            LEFTWRIST = posturePoints[LEFTWRIST,:].tolist(),
            RIGHTWRIST = posturePoints[RIGHTWRIST,:].tolist(),
            LEFTSHOULDER = posturePoints[LEFTSHOULDER,:].tolist(),
            RIGHTSHOULDER = posturePoints[RIGHTSHOULDER,:].tolist()
        )
    )

    # write results
    with open(resultsFile, 'w') as f:
        f.write(yaml.dump(results))

if __name__=='__main__':
    main()

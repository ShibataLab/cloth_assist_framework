#!/usr/bin/env python

# control.py: utility functions to execute robot policy
# Author: Nishanth Koganti
# Date: 2017/03/02
# Source: baxter_examples/scripts/joint_position_file_playback.py

# import external libraries
import sys
import time
import rospy
import numpy as np
import baxter_interface
import cPickle as pickle

# function to clean data from CSV file and return joint commands
def procLine(line, names):
    # join the joint angle values with names
    combined = zip(names[1:], line[1:])

    # convert it to a dictionary i.e. valid commands
    command = dict(combined)
    leftCommand = dict((key, command[key]) for key in command.keys() if key[:-2] == 'left_')
    rightCommand = dict((key, command[key]) for key in command.keys() if key[:-2] == 'right_')

    # return values
    return leftCommand, rightCommand

# function to read through CSV file and play data
def playFile(data, keys, threshMode=0, fThresh=None,
             forceThresh=2.0, bufferLength=10):
    """Loops through given CSV File"""

    # initialize left, right objects from Limb class
    armLeft = baxter_interface.Limb('left')
    armRight = baxter_interface.Limb('right')

    armLeft.set_joint_position_speed(0.5)
    armRight.set_joint_position_speed(0.5)

    # initialize rate object from rospy Rate class
    rate = rospy.Rate(100)

    # create numpy array for forces
    fThresh = pickle.load(open(threshName, 'rb'))
    fData = {'left':np.zeros(data.shape[0]),'right':np.zeros(data.shape[0])}

    # move to start position and start time variable
    print("[Baxter] Moving to Start Position")
    lcmdStart, rcmdStart = procLine(data[0,:], keys)
    armLeft.move_to_joint_positions(lcmdStart)
    armRight.move_to_joint_positions(rcmdStart)
    startTime = rospy.get_time()

    # create buffers for left and right force
    leftBuffer = []
    rightBuffer = []
    nSamples = data.shape[0]

    # play trajectory
    threshInd = nSamples-1
    for i in range(nSamples):
        sys.stdout.write("\r Record %d of %d " % (i, nSamples-1))
        sys.stdout.flush()

        # obtain the end effector efforts
        fL = armLeft.endpoint_effort()['force']
        fR = armRight.endpoint_effort()['force']
        fLeftRaw = np.linalg.norm([fL.x,fL.y,fL.z])
        fRightRaw = np.linalg.norm([fR.x,fR.y,fR.z])

        # append to buffer and compute moving average
        leftBuffer.append(fLeftRaw)
        rightBuffer.append(fRightRaw)
        if i >= bufferLength:
            leftBuffer.pop(0)
            rightBuffer.pop(0)

        forceLeft = np.asarray(leftBuffer).mean()
        forceRight = np.asarray(rightBuffer).mean()
        fData['left'][i] = forceLeft
        fData['right'][i] = forceRight

        # check for force thresholds
        if threshMode and (forceLeft > fThresh['left'][i,0]+forceThresh or
                           forceRight > fThresh['right'][i,0]+forceThresh):
            print "Error!! Force threshold exceed Left:%f, Right:%f" % (forceLeft, forceRight)
            threshInd = i
            break

        # parse line for commands
        lcmd, rcmd = procLine(data[i,:], keys)

        # execute these commands if the present time fits with time stamp
        # important implementation detail
        while (rospy.get_time() - startTime) < data[i,0]:
            # if we get the shutdown signal exit function
            if rospy.is_shutdown():
                print("[Baxter] Aborting - ROS shutdown")
                return False

            # execute left arm command
            if len(lcmd):
                armLeft.set_joint_positions(lcmd)
            if len(rcmd):
                armRight.set_joint_positions(rcmd)

            # sleep command
            rate.sleep()

    # return True if there is clean exit
    print
    return threshInd, fData

def rewindFile(data, keys, threshInd):
    # initialize left, right objects from Limb class
    armLeft = baxter_interface.Limb('left')
    armRight = baxter_interface.Limb('right')

    armLeft.set_joint_position_speed(0.5)
    armRight.set_joint_position_speed(0.5)

    # initialize rate object from rospy Rate class
    rate = rospy.Rate(100)

    print("[Baxter] Rewinding to Start Position")

    startTime = rospy.get_time()
    tTrack = data[threshInd,0] - data[:threshInd+1,0]

    for i in range(threshInd,-1,-1):
        sys.stdout.write("\r Record %d of %d " % (i, threshInd))
        sys.stdout.flush()

        # parse line for commands
        lcmd, rcmd = procLine(data[i,:], keys)
        while (rospy.get_time() - startTime) < tTrack[i]:
            # if we get the shutdown signal exit function
            if rospy.is_shutdown():
                print("[Baxter] Aborting - ROS shutdown")
                return False

            # execute left arm command
            if len(lcmd):
                armLeft.set_joint_positions(lcmd)
            if len(rcmd):
                armRight.set_joint_positions(rcmd)

            # sleep command
                rate.sleep()

    print
    return True

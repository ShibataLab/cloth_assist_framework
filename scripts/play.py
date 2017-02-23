#!/usr/bin/env python

# play.py: python code to play baxter recorded trajectories
# Requirements: Baxter SDK installed and connection to Baxter robot
# Author: Nishanth Koganti
# Date: 2017/02/20
# Source: baxter_examples/scripts/joint_position_file_playback.py

# import external libraries
import os
import sys
import zmq
import time
import rospy
import argparse
import numpy as np
import baxter_interface

forceThresh = 15
bufferLength = 10

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
def mapFile(data, keys):
    """Loops through given CSV File"""

    # initialize left, right objects from Limb class
    armLeft = baxter_interface.Limb('left')
    armRight = baxter_interface.Limb('right')

    armLeft.set_joint_position_speed(0.5)
    armRight.set_joint_position_speed(0.5)

    # initialize rate object from rospy Rate class
    rate = rospy.Rate(100)

    # move to start position and start time variable
    print("[Baxter] Moving to Start Position")
    lcmdStart, rcmdStart = procLine(data[0,:], keys)
    armLeft.move_to_joint_positions(lcmdStart)
    armRight.move_to_joint_positions(rcmdStart)
    startTime = rospy.get_time()

    # create buffers for left and right force
    leftBuffer = []
    rightBuffer = []

    # play trajectory
    threshInd = 0
    nSamples = data.shape[0]
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

        # check for force thresholds
        if forceLeft > forceThresh or forceRight > forceThresh:
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
    return threshInd

def rewindFile(data,keys,threshInd):
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

    return True

# main program
def main():
    """Joint Position File Playback"""

    # initialize argument parser
    argFmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class = argFmt, description = main.__doc__)

    # add arguments to parser
    parser.add_argument('-f', '--fileName', type = str, help = 'Output Joint Angle Filename')
    parser.add_argument('-t', '--thresh', type = float, help = 'Force Threshold for fail detect', )

    # parsing arguments
    args = parser.parse_args(rospy.myargv()[1:])

    # initialize node with a unique name
    print("[Baxter] Initializing Node")
    rospy.init_node("PositionFilePlayback")

    # get robot state
    rs = baxter_interface.RobotEnable()
    initState = rs.state().enabled

    # define function for clean exitprint keys

    def cleanShutdown():
        print("[Baxter] Exiting example")
        if not initState:
            print("[Baxter] Disabling Robot")
            rs.disable()

    # setup the on_shutdown program
    rospy.on_shutdown(cleanShutdown)

    # enable Robot
    print("[Baxter] Enabling Robot")
    rs.enable()

    # set the force threshold if given
    if args.thresh:
        forceThresh = args.thresh

    # if optional argument is given then only play mode is run
    if args.fileName:
        # open and read file
        data = np.genfromtxt(args.fileName, delimiter=',', names=True)
        keys = list(data.dtype.names)
        data = data.view(np.float).reshape(data.shape + (-1,))
        threshInd = mapFile(data, keys)

        time.sleep(2)
        if threshInd > 0:
            rewindFile(data, keys, threshInd)

    # if no arguments are given then it will run in sync mode
    else:
        context = zmq.Context()

        socket = context.socket(zmq.PAIR)

        socket.bind("tcp://*:5556")

        serverFlag = True
        while serverFlag:
            msg = socket.recv()
            if msg == "StopServer":
                print("[ZMQ] Received StopServer")
                serverFlag = False
                break

            elif msg == "NewTrial":
                print("[ZMQ] Recieved NewTrial")

                socket.send("Ready")
                print("[ZMQ] Sent Ready")

                msg = socket.recv()
                playbackFilename = msg
                print("[ZMQ] Received Playback Filename")

                # wait for start recording signal
                msg = socket.recv()
                if msg == "StartPlaying":
                    print("[ZMQ] Received StartPlaying")
                    playing = True
                    time.sleep(0.2)

                # start the mapFile function
                if playing:
                    data = np.genfromtxt(playbackFilename, delimiter=',', names=True)
                    keys = data.dtype.names
                    data = data.view(np.float).reshape(data.shape + (-1,))
                    threshInd = mapFile(data, keys)

                # send the stop playing message
                socket.send("StoppedPlaying")
                print "[ZMQ] Sent StoppedPlaying"

                time.sleep(2)
                if threshInd > 0:
                    rewindFile(data, keys, threshInd)

        # finish
        socket.close()

    # clean shutdown
    cleanShutdown()
    print("[Baxter] Done")

if __name__ == '__main__':
    main()

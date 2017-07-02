#!/usr/bin/env python

# play.py: python code to play baxter recorded trajectories
# Requirements: Baxter SDK installed and connection to Baxter robot
# Author: Nishanth Koganti
# Date: 2017/02/20
# Source: baxter_examples/scripts/joint_position_file_playback.py

# import external libraries
import zmq
import time
import rospy
import argparse
import numpy as np
import baxter_interface
from rl.control import playFile, rewindFile

# main program
def main():
    """Joint Position File Playback"""

    # initialize argument parser
    argFmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class = argFmt, description = main.__doc__)

    # add arguments to parser
    parser.add_argument('-f', '--fileName', type = str, help = 'Output Joint Angle Filename')
    parser.add_argument('-t', '--thresh', type = float, default = 0,
                        help = 'Force Threshold for fail detect', )

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
        playFile(data, keys, threshMode=args.thresh, fThresh=10)

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

                # start the playFile function
                if playing:
                    data = np.genfromtxt(playbackFilename, delimiter=',', names=True)
                    keys = data.dtype.names
                    data = data.view(np.float).reshape(data.shape + (-1,))
                    playFile(data, keys, threshMode=args.thresh, fThresh=15)

                # send the stop playing message
                socket.send("StoppedPlaying")
                print "[ZMQ] Sent StoppedPlaying"

        # finish
        socket.close()

    # clean shutdown
    cleanShutdown()
    print("[Baxter] Done")

if __name__ == '__main__':
    main()

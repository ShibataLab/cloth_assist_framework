// extractor.h: class definition for processing point cloud data
// Requirements: rosbag file as input
// Author: Nishanth Koganti
// Date: 2017/2/18

#pragma once
#ifndef __EXTRACTOR_H__
#define __EXTRACTOR_H__

// CPP headers
#include <cmath>
#include <string>
#include <vector>
#include <stdio.h>
#include <sstream>
#include <stdlib.h>
#include <iostream>
#include <algorithm>

// ROS headers
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rosbag/query.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

// OpenCV headers
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

// PCL headers
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>

// kinect2 bridge header
#include <kinect2_bridge/kinect2_definitions.h>

// Preprocessor directives
#define FPS 15
#define WINDOWSIZE 250

// Tracker class
class Extractor
{
  public:
    // class constructor
    Extractor(std::string fileName, std::string topicColor, std::string topicDepth, std::string topicType, std::string topicCameraInfo, bool videoMode, bool cloudMode, bool calibrateMode);

    // class destructor
    ~Extractor();

    // run function
    void run();

  private:
    // flags to save data
    int m_frame;
    bool m_videoMode, m_cloudMode, m_trackMode, m_calibrateMode;

    // write descriptor
    std::ofstream m_tracks;

    // rosbag variables
    ros::Time m_time;
    rosbag::View *m_view;
    rosbag::Bag m_bag, m_cloudBag;
    sensor_msgs::CameraInfo::ConstPtr m_cameraInfo;

    // opencv parameters
    cv::Point m_origin;
    bool m_selectObject;
    int m_width, m_height;
    std::string m_fileName;
    cv::Rect m_window, m_selection;
    cv::VideoWriter m_depthWriter, m_colorWriter;
    cv::Mat m_lookupY, m_lookupX, m_cameraMatrix;
    cv::Mat m_color, m_depth, m_output, m_points;

    // pcl initialization
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud;

    // function to create lookup table for obtaining x,y,z values
    void createLookup();

    // function to display images
    void cloudExtract();

    // function to obtain cloth calibration values
    void windowCalibrate();

    // function to obtain camera info from message filter msgs
    void readCameraInfo();

    // function to get ROI from color and depth images
    void createROI(cv::Mat &roi);

    // function to build point cloud from roi
    void createCloud(cv::Mat &roi);

    // function to obtain cv::Mat from sensor_msgs
    void readImage(sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image);

    // function to downscale image
    void dispDepth(const cv::Mat &in, cv::Mat &out, const float maxValue);

    // mouse click callback function for T-shirt color calibration
    static void onMouse(int event, int x, int y, int flags, void* param);
};

#endif

// extractor.cpp: Class implementation to read rosbag files and obtain T-shirt point cloud
// Requirements: rosbag file as input
// Author: Nishanth Koganti
// Date: 2017/2/18

#include <extractor.h>

// class constructor
Extractor::Extractor(std::string fileName, std::string topicColor, std::string topicDepth, std::string topicCameraInfo, std::string topicType, bool videoMode, bool cloudMode, bool calibrateMode)
  : m_videoMode(videoMode), m_cloudMode(cloudMode), m_calibrateMode(calibrateMode)
{
  // initialize select object flag
  m_trackMode = true;
  m_selectObject = false;

  // create matrices for intrinsic parameters
  m_cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);

  // create char file names
  char colorVideoName[200], depthVideoName[200];
  char bagName[200], cloudBagName[200], tracksName[200];
  // open the bag file
  sprintf(bagName, "%s.bag", fileName.c_str());
  m_bag.open(bagName, rosbag::bagmode::Read);

  // create vector of topics for querying
  std::vector<std::string> topics;
  topics.push_back(topicColor);
  topics.push_back(topicDepth);
  topics.push_back(topicCameraInfo);

  // create view instance for rosbag parsing
  m_view = new rosbag::View(m_bag, rosbag::TopicQuery(topics));

  // set the width and height parameters for cloth tracking functions
  if(topicType == "hd")
  {
    m_height = 1080;
    m_width = 1920;
  }
  else if(topicType == "qhd")
  {
    m_height = 540;
    m_width = 960;
  }
  else if(topicType == "sd")
  {
    m_height = 424;
    m_width = 512;
  }

  m_fileName = fileName;

  // trackMode
  if (m_trackMode)
  {
    sprintf(tracksName, "%s", fileName.c_str());
    m_tracks.open(tracksName);
    m_tracks << "Frame,Time" << std::endl;
  }

  // cloudMode
  if (m_cloudMode)
  {
    sprintf(cloudBagName, "%sCloud.bag", fileName.c_str());
    m_cloudBag.open(cloudBagName, rosbag::bagmode::Write);
  }

  // videoMode
  if (m_videoMode)
  {
    sprintf(colorVideoName, "%sColor.avi", fileName.c_str());
    sprintf(depthVideoName, "%sDepth.avi", fileName.c_str());
    m_colorWriter.open (colorVideoName, CV_FOURCC('D','I','V','X'), FPS, cv::Size (WINDOWSIZE,WINDOWSIZE), true);
    m_depthWriter.open (depthVideoName, CV_FOURCC('D','I','V','X'), FPS, cv::Size (WINDOWSIZE,WINDOWSIZE), false);
  }
}

// class destructor
Extractor::~Extractor()
{
}

// write run function
void Extractor::run()
{
  // opencv variables
  cv::Mat roi;
  cv::namedWindow("Depth",1);
  cv::namedWindow("Color", 1);

  // start parsing rosbag
  std::string type;
  std::string topicCloud = "/cloth/cloud";
  std::string topicColor = "/cloth/color";
  std::string topicDepth = "/cloth/depth";

  cv_bridge::CvImage sColor,sDepth;
  rosbag::View::iterator iter = m_view->begin();

  double currentTime, timeTrack, startTime;

  // create pcl cloud viewer
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("Cloth"));
  viewer->setCameraPosition(-0.0184,0.1334,0.0424,-0.2173,0.2068,2.2920,-0.0836,-0.9961,0.0251);
  viewer->setSize(600,600);

  // ros time instance
  m_time = (*iter).getTime();
  startTime = m_time.toSec();

  // read first set of images for calibration
  for (int i = 0; i < 3; i++)
  {
    rosbag::MessageInstance const m = *iter;

    type = m.getDataType();
    if (type == "sensor_msgs/Image")
    {
      sensor_msgs::Image::ConstPtr image = m.instantiate<sensor_msgs::Image>();
      if (image->encoding == "bgr8")
        readImage(image, m_color);
      else if (image->encoding == "16UC1")
        readImage(image, m_depth);
    }
    else
      m_cameraInfo = m.instantiate<sensor_msgs::CameraInfo>();

    ++iter;
  }

  // perform window calibration
  windowCalibrate();

  // main loop
  m_frame = 1;
  while(iter != m_view->end())
  {
    m_time = (*iter).getTime();
    currentTime = m_time.toSec();
    timeTrack = currentTime - startTime;

    for (int i = 0; i < 3; i++)
    {
      rosbag::MessageInstance const m = *iter;

      type = m.getDataType();
      if (type == "sensor_msgs/Image")
      {
        sensor_msgs::Image::ConstPtr image = m.instantiate<sensor_msgs::Image>();
        if (image->encoding == "bgr8")
          readImage(image, m_color);
        else if (image->encoding == "16UC1")
          readImage(image, m_depth);
      }
      else
        m_cameraInfo = m.instantiate<sensor_msgs::CameraInfo>();

      ++iter;
    }

    cloudExtract();

    if (m_trackMode)
      m_tracks << m_frame << "," << timeTrack << std::endl;

    if (m_videoMode)
    {
      m_colorWriter.write(m_output);
      m_depthWriter.write(m_points);
    }

    if (m_cloudMode)
    {
      sColor.header.stamp = m_time;
      sColor.header.frame_id = "color";
      sColor.encoding = "8UC1"; sColor.image = m_output;

      sDepth.header.stamp = m_time;
      sDepth.header.frame_id = "depth";
      sDepth.encoding = "8UC1"; sDepth.image = m_points;

      m_cloudBag.write(topicColor, m_time, sColor);
      m_cloudBag.write(topicDepth, m_time, sDepth);
      m_cloudBag.write(topicCloud, m_time, *m_cloud);
    }

    cv::imshow("Depth", m_points);
    cv::imshow("Color", m_output);

    if (m_frame == 1)
      viewer->addPointCloud(m_cloud,"cloth");
    else
      viewer->updatePointCloud(m_cloud,"cloth");
    viewer->spinOnce(33);

    cv::waitKey(20);
    m_frame++;
  }

  // clean exit
  m_bag.close();
  if (m_videoMode)
  {
    m_colorWriter.release();
    m_depthWriter.release();
  }

  if (m_cloudMode)
    m_cloudBag.close();

}

// function to obtain cloth calibration values
void Extractor::windowCalibrate()
{
	// opencv initialization
	if (m_calibrateMode)
  {
    char key = 0;
	  cv::Mat disp;

	  // gui initialization
	  cv::namedWindow("WindowCalibrate", CV_WINDOW_AUTOSIZE);
	  cv::setMouseCallback("WindowCalibrate", onMouse, (void *)this);

	  // create a copy of color image
    m_color.copyTo(disp);

    // inifinite for loop
    while(ros::ok())
    {
		  // show the selection
		  if (m_selectObject)
		  {
			  m_color.copyTo(disp);
			  cv::Mat roi(disp, m_selection);
			  cv::bitwise_not(roi, roi);
		  }

		  // display the image
		  cv::imshow("WindowCalibrate", disp);
		  key = cv::waitKey(5);
		  if (key == 'q')
			  break;
	  }

	  // destroy the window
    m_window = m_selection;
	  cv::destroyWindow("WindowCalibrate");
    std::cout << m_window.x << " " << m_window.y << endl;
  }
  else
  {
    m_window = cv::Rect(116, 93, WINDOWSIZE, WINDOWSIZE);
  }
}

// function to display images
void Extractor::cloudExtract()
{
  // variable initialization
  cv::Mat roi;

  // function to extract image roi from color and depth functions
  createROI(roi);

  // function to create point cloud and obtain
  createCloud(roi);
}

// function to get image roi from color and depth images
void Extractor::createROI(cv::Mat &roi)
{
  // opencv initialization
  cv::Mat temp;

  roi = m_depth(m_window);
  m_output = m_color(m_window);

  roi.copyTo(temp);
  dispDepth(temp, m_points, 4096.0f);
}

// function to create point cloud from extracted ROI
void Extractor::createCloud(cv::Mat &roi)
{
  // initialize cloud
  m_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());

  // set cloud parameters
  m_cloud->header.frame_id = "cloth_frame";
  m_cloud->width = roi.cols;
  m_cloud->height = roi.rows;

  m_cloud->is_dense = true;
  m_cloud->points.resize(m_cloud->height * m_cloud->width);

  // create lookup tables
  readCameraInfo();
  createLookup();

  // variables
  const float badPoint = std::numeric_limits<float>::quiet_NaN();

  int xOffset, yOffset;
  xOffset = m_window.x; yOffset = m_window.y;

  // parallel processing of pixel values
  #pragma omp parallel for
  for(int r = 0; r < roi.rows; ++r)
  {
    // create row of Points
    pcl::PointXYZ *itP = &m_cloud->points[r * roi.cols];

    // get pointer to row in depth image
    const uint16_t *itD = roi.ptr<uint16_t>(r);

    // get the x and y values
    const float y = m_lookupY.at<float>(0, yOffset+r);
    const float *itX = m_lookupX.ptr<float>();
    itX = itX + xOffset;

    // convert all the depth values in the depth image to Points in point cloud
    for(size_t c = 0; c < (size_t)roi.cols; ++c, ++itP, ++itD, ++itX)
    {
      register const float depthValue = *itD / 1000.0f;

      // Check for invalid measurements
      if(isnan(depthValue) || depthValue <= 0.1)
      {
        // set values to NaN for later processing
        itP->x = itP->y = itP->z = badPoint;
        continue;
      }

      // set the values for good points
      itP->z = depthValue;
      itP->x = *itX * depthValue;
      itP->y = y * depthValue;
    }
  }
}

// mouse click callback function for T-shirt color calibration
void Extractor::onMouse(int event, int x, int y, int flags, void* param)
{
  // this line needs to be added if we want to access the class private parameters
  // within a static function
  // URL: http://stackoverflow.com/questions/14062501/giving-callback-function-access-to-class-data-members-in-c
  Extractor* ptr = static_cast<Extractor*>(param);

	if (ptr->m_selectObject)
	{
    ptr->m_selection.width = WINDOWSIZE;
		ptr->m_selection.height = WINDOWSIZE;
    ptr->m_selection.x = std::min(x, ptr->m_origin.x);
		ptr->m_selection.y = std::min(y, ptr->m_origin.y);
		ptr->m_selection &= cv::Rect(0, 0, ptr->m_width, ptr->m_height);
	}

	switch (event)
	{
	case cv::EVENT_LBUTTONDOWN:
		ptr->m_origin = cv::Point(x, y);
		ptr->m_selection = cv::Rect(x, y, 0, 0);
		ptr->m_selectObject = true;
		break;
	case cv::EVENT_LBUTTONUP:
		ptr->m_selectObject = false;
		break;
	}
}

void Extractor::dispDepth(const cv::Mat &in, cv::Mat &out, const float maxValue)
{
  out = cv::Mat(in.rows, in.cols, CV_8U);
  const uint32_t maxInt = 255;

  #pragma omp parallel for
  for(int r = 0; r < in.rows; ++r)
  {
    const uint16_t *itI = in.ptr<uint16_t>(r);
    uint8_t *itO = out.ptr<uint8_t>(r);

    for(int c = 0; c < in.cols; ++c, ++itI, ++itO)
    {
      *itO = (uint8_t)std::min((*itI * maxInt / maxValue), 255.0f);
    }
  }
}

void Extractor::readImage(sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image)
{
  // obtain image data and encoding from sensor msg
  cv_bridge::CvImageConstPtr pCvImage;
  pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);

  // copy data to the Mat image
  pCvImage->image.copyTo(image);
}

// function to obtain camera info from message filter msgs
void Extractor::readCameraInfo()
{
  // get pointer for first element in cameraMatrix
  double *itC = m_cameraMatrix.ptr<double>(0, 0);

  // create for loop to copy complete data
  for(size_t i = 0; i < 9; ++i, ++itC)
  {
    *itC = m_cameraInfo->K[i];
  }
}

// function to create lookup table for obtaining x,y,z values
void Extractor::createLookup()
{
  // get the values from the camera matrix of intrinsic parameters
  const float fx = 1.0f / m_cameraMatrix.at<double>(0, 0);
  const float fy = 1.0f / m_cameraMatrix.at<double>(1, 1);
  const float cx = m_cameraMatrix.at<double>(0, 2);
  const float cy = m_cameraMatrix.at<double>(1, 2);

  // float iterator
  float *it;

  // lookup table for y pixel locations
  m_lookupY = cv::Mat(1, m_height, CV_32F);
  it = m_lookupY.ptr<float>();
  for(size_t r = 0; r < m_height; ++r, ++it)
  {
    *it = (r - cy) * fy;
  }

  // lookup table for x pixel locations
  m_lookupX = cv::Mat(1, m_width, CV_32F);
  it = m_lookupX.ptr<float>();
  for(size_t c = 0; c < m_width; ++c, ++it)
  {
    *it = (c - cx) * fx;
  }
}

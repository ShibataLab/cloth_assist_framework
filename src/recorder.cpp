// recorder.cpp: Class definition for recording kinect2 topics than can be processed later
// Requirements: recorder class definition header
// Author: Nishanth Koganti
// Date: 2017/02/21

// all required header files included in the class definition header
#include <recorder.h>

// class constructor
Recorder::Recorder(const std::string &topicColor, const std::string &topicDepth, const int recordMode)
    : topicColor(topicColor), topicDepth(topicDepth), frame(0), queueSize(5), nh("~"), spinner(0), it(nh), recordMode(recordMode)
{
}

// class destructor
Recorder::~Recorder()
{
}

// run function
void Recorder::run()
{
  // zmq variables
  char flag;
  std::string Message;

  // start the kinect depth sensor
  start();

  if (recordMode == 0)
  {
    // fileName set to default in basic record mode
    // to make my life easier
    fileName = "default";

    // open a new bag file
    char bagName[200];
    sprintf(bagName, "%s.bag", fileName.c_str());
    bag.open(bagName, rosbag::bagmode::Write);
    std::cout << "[ROSBAG] Opening ROS bag" << std::endl;

    std::cout << "Start recording? Y/N" << std::endl;
    std::cin >> flag;

    // start the recording
    spinner.start();

    std::cout << "Stop recording? Y/N" << std::endl;
    std::cin >> flag;

    // stop the spinner
    spinner.stop();

    // close rosbag
    bag.close();
    std::cout << "[ROSBAG] Saving ROS Bag" << std::endl;
  }
  else if (recordMode == 1)
  {
    // initialize zmq variables
    zmq::context_t context(1);

    // initialize socket with the pair option
    zmq::socket_t kinectSocket(context, ZMQ_PAIR);

    // create kinect server to synchronize with
    kinectSocket.bind("tcp://*:5555");
    std::cout << "[ZMQ] Created kinect server" << std::endl;

    while (1)
    	{
        // wait for message from clinet
    		zmq::message_t request;
    		kinectSocket.recv(&request);
    		Message = std::string(static_cast<char *>(request.data()), request.size());

        // stop server and clean exit if Stop message is received
        if (Message == "StopServer")
    		{
    			std::cout << "[ZMQ] Received Stop Server" << std::endl;
    			break;
    		}

        // initialize variables for recording if New message is received
        else if (Message == "NewTrial")
    		{
    			std::cout << "[ZMQ] Received NewTrial" << std::endl;

    			// send ready reply
    			zmq::message_t reply(5);
    			std::memcpy((void *)reply.data(), "Ready", 5);
    			kinectSocket.send(reply);
    			std::cout << "[ZMQ] Sent Ready" << std::endl;

    			// get kinect filename
    			zmq::message_t recordName;
    			kinectSocket.recv(&recordName);
    			fileName = std::string(static_cast<char *>(recordName.data()), recordName.size());
    			std::cout << "[ZMQ] Received filename" << std::endl;

          // open a new bag file
          char bagName[200];
          sprintf(bagName, "%s.bag", fileName.c_str());
          bag.open(bagName, rosbag::bagmode::Write);
          std::cout << "[ROSBAG] Opening ROS bag" << std::endl;

          // wait for start recording message from client
          zmq::message_t startRequest;
    			kinectSocket.recv(&startRequest);
    			Message = std::string(static_cast<char *>(startRequest.data()), startRequest.size());
          if (Message == "StartRecording")
          {
            std::cout << "[ZMQ] Received StartRecording" << std::endl;

            // start the recording
            spinner.start();
          }
    			else
          {
            std::cout << "[ZMQ] Invalid message" << std::endl;
            continue;
          }

          // wait for stop recording message from client
      		zmq::message_t stopRequest;
      		kinectSocket.recv(&stopRequest);
      		Message = std::string(static_cast<char *>(stopRequest.data()), stopRequest.size());
          if (Message == "StopRecording")
          {
            std::cout << "[ZMQ] Received StopRecording" << std::endl;
          }
    			else
          {
            std::cout << "[ZMQ] Message Received: " << Message << std::endl;
            std::cout << "[ZMQ] Invalid message" << std::endl;
          }

          // stop the spinner
          spinner.stop();

          // close rosbag
          bag.close();
          std::cout << "[ROSBAG] Saving ROS Bag" << std::endl;
    		}
    	}

    // clean shutdown
    kinectSocket.close();
  }

  // stop function
  stop();
}

// start function
void Recorder::start()
{
  // get the camera info ros topics for color and depth
  topicCameraInfoColor = topicColor.substr(0, topicColor.rfind('/')) + "/camera_info";
  topicCameraInfoDepth = topicDepth.substr(0, topicDepth.rfind('/')) + "/camera_info";

  // TransportHints stores the transport settings for the image topic subscriber
  // here we are giving the setting of raw or compressed using the useCompressed variable
  image_transport::TransportHints hints("raw");

  // SubscriberFilters are used to subscribe to the kinect image topics
  subImageColor = new image_transport::SubscriberFilter(it, topicColor, queueSize, hints);
  subImageDepth = new image_transport::SubscriberFilter(it, topicDepth, queueSize, hints);

  // message filters are used to subscribe to the camera info topics
  subCameraInfoColor = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoColor, queueSize);
  subCameraInfoDepth = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoDepth, queueSize);

  // creating a exact synchronizer for 4 ros topics with queueSize
  // the recorder class callback function is set as the callback function
  syncExact = new message_filters::Synchronizer<ExactSyncPolicy>(ExactSyncPolicy(queueSize), *subImageColor, *subImageDepth, *subCameraInfoColor, *subCameraInfoDepth);
  syncExact->registerCallback(boost::bind(&Recorder::callback, this, _1, _2, _3, _4));
}

// stop function to have clean shutdown
void Recorder::stop()
{
  // clean up all variables
  delete syncExact;

  delete subImageColor;
  delete subImageDepth;
  delete subCameraInfoColor;
  delete subCameraInfoDepth;
}

// message filter callback function, all processing is done here
void Recorder::callback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
              const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth)
{
  ros::Time time = ros::Time::now();

  // write point cloud message and camera info to the ros bag file
  bag.write(topicColor, time, *imageColor);
  bag.write(topicDepth, time, *imageDepth);
  bag.write(topicCameraInfoColor, time, *cameraInfoColor);
}

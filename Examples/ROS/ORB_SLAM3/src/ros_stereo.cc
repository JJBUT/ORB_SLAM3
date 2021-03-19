/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez
 * Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós,
 * University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ORB-SLAM3. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>

#include "../../../include/System.h"
#include "iv_slam_helpers/torch_helpers.h"
#include "std_srvs/Trigger.h"

using namespace std;

// Tracking states
enum eTrackingState {
  SYSTEM_NOT_READY = -1,
  NO_IMAGES_YET = 0,
  NOT_INITIALIZED = 1,
  OK = 2,
  RECENTLY_LOST = 3,
  LOST = 4,
  OK_KLT = 5
};

// Convert CV SE(3) mat to ROS Pose
geometry_msgs::Pose cvMatToPose(cv::Mat& cv_pose) {
  tf2::Matrix3x3 tf2_rot(cv_pose.at<float>(0, 0),
                         cv_pose.at<float>(0, 1),
                         cv_pose.at<float>(0, 2),
                         cv_pose.at<float>(1, 0),
                         cv_pose.at<float>(1, 1),
                         cv_pose.at<float>(1, 2),
                         cv_pose.at<float>(2, 0),
                         cv_pose.at<float>(2, 1),
                         cv_pose.at<float>(2, 2));

  tf2::Vector3 tf2_trans(cv_pose.at<float>(0, 3),
                         cv_pose.at<float>(1, 3),
                         cv_pose.at<float>(2, 3));

  // Create a transform and convert to a Pose
  tf2::Transform tf2_transform(tf2_rot, tf2_trans);
  geometry_msgs::Pose ros_pose;
  tf2::toMsg(tf2_transform, ros_pose);

  return ros_pose;
}

// Provide approximate and exact stereo image synchronization policies
typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image,
                                                  sensor_msgs::Image>
    ExactPolicy;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image>
    ApproximatePolicy;
typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;

class ImageGrabber {
 public:
  ImageGrabber(ros::NodeHandle& nh, ORB_SLAM3::System* pSLAM)
      : nh_(nh), mpSLAM(pSLAM) {
    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("pose", 1);
  }

  void GrabStereo(const sensor_msgs::ImageConstPtr& msgLeft,
                  const sensor_msgs::ImageConstPtr& msgRight);

  ORB_SLAM3::System* mpSLAM;
  bool do_rectify;
  bool introspection_on;
  cv::Mat M1l, M2l, M1r, M2r;

  // ROS publishing utils
  ros::NodeHandle nh_;
  ros::Publisher pose_pub_;

  // Introspection utils
  torch::jit::script::Module introspection_model;
  torch::Device device = torch::kCPU;
};

// A server that advertises a service which will call the system reset function
class ResetServer {
 public:
  ResetServer(ros::NodeHandle& nh, ORB_SLAM3::System* pSLAM)
      : nh_(nh), mpSLAM_(pSLAM) {
    reset_vslam_server_ =
        nh_.advertiseService("reset", &ResetServer::ResetServerCB, this);
  }

  bool ResetServerCB(std_srvs::Trigger::Request& req,
                     std_srvs::Trigger::Response& res);

  ros::NodeHandle nh_;
  ORB_SLAM3::System* mpSLAM_;
  ros::ServiceServer reset_vslam_server_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "Stereo");
  ros::start();

  // Load launch file parameters from parameter server
  ros::NodeHandle private_nh("~");

  std::string path_to_vocabulary;
  if (!private_nh.getParam("path_to_vocabulary", path_to_vocabulary)) {
    ROS_ERROR("Could not load parameter: 'path_to_vocabulary'");
    ros::shutdown();
    return -1;
  }
  std::string path_to_settings;
  if (!private_nh.getParam("path_to_settings", path_to_settings)) {
    ROS_ERROR("Could not load parameter: 'path_to_settings'");
    ros::shutdown();
    return -1;
  }
  std::string path_to_introspection_model;
  if (!private_nh.getParam("path_to_introspection_model",
                           path_to_introspection_model)) {
    ROS_ERROR("Could not load parameter: 'path_to_introspection_model'");
    ros::shutdown();
    return -1;
  }
  // Undistort or/and rectify - if you don't want on of these leave
  // the distortion or/and rectification parameters as zero in the config file.
  // But you do have to provide them
  bool undistort_and_rectify_on;
  if (!private_nh.getParam("undistort_and_rectify_on",
                           undistort_and_rectify_on)) {
    ROS_ERROR("Could not load parameter: 'undistort_and_rectify_on'");
    ros::shutdown();
    return -1;
  }
  bool introspection_on;
  if (!private_nh.getParam("introspection_on", introspection_on)) {
    ROS_ERROR("Could not load parameter: 'introspection_on'");
    ros::shutdown();
    return -1;
  }
  bool gpu_available;
  if (!private_nh.getParam("gpu_available", gpu_available)) {
    ROS_ERROR("Could not load parameter: 'gpu_available'");
    ros::shutdown();
    return -1;
  }
  // If approximate sync if on then an approximate time filter is used which is
  // useful when the image pairs dont have the exact same time stamp
  bool approximate_sync_on;
  if (!private_nh.getParam("approximate_sync_on", approximate_sync_on)) {
    ROS_ERROR("Could not load parameter: 'approximate_sync_on'");
    ros::shutdown();
    return -1;
  }
  string image_transport_type;
  if (!private_nh.getParam("image_transport_type", image_transport_type)) {
    ROS_ERROR("Could not load parameter: 'image_transport_type'");
    ros::shutdown();
    return -1;
  }

  // Create SLAM system. It initializes all system threads and gets ready to
  // process frames.
  ORB_SLAM3::System SLAM(
      path_to_vocabulary, path_to_settings, ORB_SLAM3::System::STEREO, true);

  ResetServer rs(private_nh, &SLAM);

  ImageGrabber igb(private_nh, &SLAM);

  igb.do_rectify = undistort_and_rectify_on;
  igb.introspection_on = introspection_on;

  // Load introspection model
  // torch::jit::script::Module introspection_model;
  // torch::Device device = torch::kCPU;
  if (introspection_on) {
    // Check if we have a GPU to run on
    if (gpu_available && torch::cuda::is_available()) {
      igb.device = torch::kCUDA;
      ROS_INFO("Introspection model running on GPU :)");
    } else {
      ROS_WARN("Introspection model running on CPU :(");
    }
    try {
      // Deserialize the ScriptModule from file
      igb.introspection_model = torch::jit::load(path_to_introspection_model);
      igb.introspection_model.to(igb.device);
    } catch (const c10::Error& e) {
      ROS_ERROR("Error deserializing the ScriptModule from file");
      ros::shutdown();
      return -1;
    }
  }

  if (igb.do_rectify) {
    // Load settings related to stereo calibration
    cv::FileStorage fsSettings(path_to_settings, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
      ROS_ERROR("ERROR: Wrong path to settings");
      ros::shutdown();
      return -1;
    }

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() ||
        R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
        rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0) {
      ROS_ERROR("ERROR: Calibration parameters to rectify stereo are missing!");
      ros::shutdown();
      return -1;
    }

    cv::initUndistortRectifyMap(K_l,
                                D_l,
                                R_l,
                                P_l.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_l, rows_l),
                                CV_32F,
                                igb.M1l,
                                igb.M2l);
    cv::initUndistortRectifyMap(K_r,
                                D_r,
                                R_r,
                                P_r.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_r, rows_r),
                                CV_32F,
                                igb.M1r,
                                igb.M2r);
  }

  // Subscribe to image topics using image tranport. A transport hint and
  // specification of approximate/exact time sync policy are required
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::TransportHints hints(image_transport_type);
  image_transport::SubscriberFilter left_sub(
      it, "/stereo/left/image_raw", 1, hints);
  image_transport::SubscriberFilter right_sub(
      it, "/stereo/right/image_raw", 1, hints);

  ExactSync exact_sync(ExactPolicy(10), left_sub, right_sub);
  ApproximateSync approximate_sync(ApproximatePolicy(10), left_sub, right_sub);

  if (approximate_sync_on) {
    approximate_sync.registerCallback(
        boost::bind(&ImageGrabber::GrabStereo, &igb, _1, _2));
  } else {
    // Time stamps will have to match exactly
    exact_sync.registerCallback(
        boost::bind(&ImageGrabber::GrabStereo, &igb, _1, _2));
  }

  ros::spin();

  // Stop all threads
  SLAM.Shutdown();

  // Save camera trajectory
  SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_TUM_Format.txt");
  SLAM.SaveTrajectoryTUM("FrameTrajectory_TUM_Format.txt");
  SLAM.SaveTrajectoryKITTI("FrameTrajectory_KITTI_Format.txt");

  ros::shutdown();

  return 0;
}

void ImageGrabber::GrabStereo(const sensor_msgs::ImageConstPtr& msgLeft,
                              const sensor_msgs::ImageConstPtr& msgRight) {
  // Copy the ros image message to cv::Mat.
  cv_bridge::CvImageConstPtr cv_ptrLeft;
  try {
    cv_ptrLeft = cv_bridge::toCvShare(msgLeft);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv_bridge::CvImageConstPtr cv_ptrRight;
  try {
    cv_ptrRight = cv_bridge::toCvShare(msgRight);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // Rectify and undistort images if needed
  cv::Mat imLeft, imRight;
  if (this->do_rectify) {
    cv::remap(cv_ptrLeft->image, imLeft, M1l, M2l, cv::INTER_LINEAR);
    cv::remap(cv_ptrRight->image, imRight, M1r, M2r, cv::INTER_LINEAR);
  } else {
    // Don't undistort/rectify
    imLeft = cv_ptrLeft->image;
    imRight = cv_ptrRight->image;
  }

  // Feed left image to model to create cost mask
  cv::Mat cost_img_cv;
  if (this->introspection_on) {
    // Run inference on the introspection model online
    cv::Mat imLeft_RGB =
        imLeft;  // TODO initializae imLeft_RGB as blank instead of imLeft
    cv::cvtColor(imLeft_RGB, imLeft_RGB, CV_BGR2RGB);

    // Convert to float and normalize image
    imLeft_RGB.convertTo(imLeft_RGB, CV_32FC3, 1.0 / 255.0);
    cv::subtract(imLeft_RGB,
                 cv::Scalar(0.485, 0.456, 0.406),
                 imLeft_RGB);  // TODO what are these numbers
    cv::divide(imLeft_RGB, cv::Scalar(0.229, 0.224, 0.225), imLeft_RGB);

    auto tensor_img = ORB_SLAM3::CVImgToTensor(imLeft_RGB);
    // Swap axis
    tensor_img = ORB_SLAM3::TransposeTensor(tensor_img, {(2), (0), (1)});
    // Add batch dim
    tensor_img.unsqueeze_(0);

    tensor_img = tensor_img.to(this->device);
    std::vector<torch::jit::IValue> inputs{tensor_img};
    at::Tensor cost_img;
    cost_img = this->introspection_model.forward(inputs).toTensor();
    cost_img = (cost_img * 255.0).to(torch::kByte);
    cost_img = cost_img.to(torch::kCPU);

    cost_img_cv = ORB_SLAM3::ToCvImage(cost_img);
  }

  // Pass the images to the SLAM system
  cv::Mat cv_pose;
  if (this->introspection_on) {
    cv_pose = mpSLAM->TrackStereo(imLeft,
                                  imRight,
                                  cv_ptrLeft->header.stamp.toSec(),
                                  this->introspection_on,
                                  cost_img_cv);
  } else {
    cv_pose =
        mpSLAM->TrackStereo(imLeft, imRight, cv_ptrLeft->header.stamp.toSec());
  }
  // Publish the pose
  geometry_msgs::PoseStamped ros_pose_stamped;
  ros_pose_stamped.pose = cvMatToPose(cv_pose);
  ros_pose_stamped.header.stamp =
      cv_ptrLeft->header.stamp;  // This time may be old by now?
  ros_pose_stamped.header.frame_id = "orb_slam";
  pose_pub_.publish(ros_pose_stamped);

  return;
}

bool ResetServer::ResetServerCB(std_srvs::Trigger::Request& req,
                                std_srvs::Trigger::Response& res) {
  if (mpSLAM_->GetTrackingState() == OK) {
    mpSLAM_->Reset();
    res.success = true;
    res.message = "Called reset_vslam_server_";
    return true;
  } else {
    // Tracking is not ok - do not call reset becuase who knows what segfaults
    // we will cause :(
    res.success = false;
    res.message =
        "Not able to call reset_vslam_server_ because tracking is not OK";
    return true;
  }
}
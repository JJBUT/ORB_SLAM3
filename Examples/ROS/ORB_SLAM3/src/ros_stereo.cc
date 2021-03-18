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
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <ros/ros.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>

#include "../../../include/System.h"

using namespace std;

class ImageGrabber {
 public:
  ImageGrabber(ORB_SLAM3::System* pSLAM) : mpSLAM(pSLAM) {}

  void GrabStereo(const sensor_msgs::ImageConstPtr& msgLeft,
                  const sensor_msgs::ImageConstPtr& msgRight);

  ORB_SLAM3::System* mpSLAM;
  bool do_rectify;
  cv::Mat M1l, M2l, M1r, M2r;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "RGBD");
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

  // Create SLAM system. It initializes all system threads and gets ready to
  // process frames.
  ORB_SLAM3::System SLAM(
      path_to_vocabulary, path_to_settings, ORB_SLAM3::System::STEREO, true);

  ImageGrabber igb(&SLAM);

  igb.do_rectify = undistort_and_rectify_on;

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

  ros::NodeHandle nh;

  message_filters::Subscriber<sensor_msgs::Image> left_sub(
      nh, "/stereo/left/image_raw", 1);
  message_filters::Subscriber<sensor_msgs::Image> right_sub(
      nh, "/stereo/right/image_raw", 1);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image>
      sync_pol;
  message_filters::Synchronizer<sync_pol> sync(
      sync_pol(10), left_sub, right_sub);
  sync.registerCallback(boost::bind(&ImageGrabber::GrabStereo, &igb, _1, _2));

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

  if (do_rectify) {
    cv::Mat imLeft, imRight;
    cv::remap(cv_ptrLeft->image, imLeft, M1l, M2l, cv::INTER_LINEAR);
    cv::remap(cv_ptrRight->image, imRight, M1r, M2r, cv::INTER_LINEAR);
    mpSLAM->TrackStereo(imLeft, imRight, cv_ptrLeft->header.stamp.toSec());
  } else {
    mpSLAM->TrackStereo(cv_ptrLeft->image,
                        cv_ptrRight->image,
                        cv_ptrLeft->header.stamp.toSec());
  }
}

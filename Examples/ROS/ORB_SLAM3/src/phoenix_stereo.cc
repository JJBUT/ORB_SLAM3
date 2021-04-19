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
#include <omnimapper_msgs/RelativePoseQuery.h>  // Phoenix service message
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

cv::Mat CalculateInverseTransform(const cv::Mat& transform);

cv::Mat CalculateRelativeTransform(const cv::Mat& dest_frame_pose,
                                   const cv::Mat& src_frame_pose);

// Convert CV SE(3) mat to ROS Pose
geometry_msgs::Pose cvMatToPose(const cv::Matx44f& cv_pose) {
  cv::Matx33f cv_rotation(cv_pose(0, 0),
                          cv_pose(0, 1),
                          cv_pose(0, 2),
                          cv_pose(1, 0),
                          cv_pose(1, 1),
                          cv_pose(1, 2),
                          cv_pose(2, 0),
                          cv_pose(2, 1),
                          cv_pose(2, 2));
  cv::Matx31f cv_translation(cv_pose(0, 3), cv_pose(1, 3), cv_pose(2, 3));

  cv::Matx33f coordinate_rotation(
      0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0);

  cv::Matx33f rotated_cv_rotation = cv_rotation;  // TODO the rotation has a bug
  cv::Matx31f rotated_cv_translation = coordinate_rotation * cv_translation;

  // Convert to ROS types
  tf2::Matrix3x3 tf2_rot(rotated_cv_rotation(0, 0),
                         rotated_cv_rotation(0, 1),
                         rotated_cv_rotation(0, 2),
                         rotated_cv_rotation(1, 0),
                         rotated_cv_rotation(1, 1),
                         rotated_cv_rotation(1, 2),
                         rotated_cv_rotation(2, 0),
                         rotated_cv_rotation(2, 1),
                         rotated_cv_rotation(2, 2));
  tf2::Vector3 tf2_trans(rotated_cv_translation(0),
                         rotated_cv_translation(1),
                         rotated_cv_translation(2));

  // Create a tf2 pose/transform and convert to a Pose
  tf2::Transform tf2_pose(tf2_rot, tf2_trans);
  geometry_msgs::Pose ros_pose;
  tf2::toMsg(tf2_pose, ros_pose);

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
  ImageGrabber(ros::NodeHandle& nh,
               std::string const& orb_slam_frame,
               ORB_SLAM3::System* pSLAM)
      : nh_(nh), orb_slam_frame_(orb_slam_frame), mpSLAM(pSLAM) {
    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("pose", 1);
  }

  void GrabStereo(const sensor_msgs::ImageConstPtr& msgLeft,
                  const sensor_msgs::ImageConstPtr& msgRight);

  // ROS publishing utils
  ros::NodeHandle nh_;
  ros::Publisher pose_pub_;
  std::string orb_slam_frame_;

  // Introspection utils
  torch::jit::script::Module introspection_model;
  torch::Device device = torch::kCPU;

  // ORB SLAM
  ORB_SLAM3::System* mpSLAM;
  bool do_rectify;
  bool introspection_on;
  cv::Mat M1l, M2l, M1r, M2r;
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

// A server that advertises a service which will call the system reset function
class OmnigraphInterface {
 public:
  OmnigraphInterface(ros::NodeHandle& nh, ORB_SLAM3::System* pSLAM)
      : nh_(nh), mpSLAM_(pSLAM) {
    omnigraph_interface_server_ = nh_.advertiseService(
        "/warty/viwo_odom", &OmnigraphInterface::OmnigraphInterfaceCB, this);
  }

  bool OmnigraphInterfaceCB(omnimapper_msgs::RelativePoseQuery::Request& req,
                            omnimapper_msgs::RelativePoseQuery::Response& res);

  ros::NodeHandle nh_;
  ORB_SLAM3::System* mpSLAM_;
  ros::ServiceServer omnigraph_interface_server_;

  cv::Matx44f last_pose = cv::Mat(cv::Mat::eye(4, 4, CV_32F));
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

  bool viewer_on;
  if (!private_nh.getParam("viewer_on", viewer_on)) {
    ROS_ERROR("Could not load parameter: 'viewer_on'");
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

  string orb_slam_frame;
  if (!private_nh.getParam("orb_slam_frame", orb_slam_frame)) {
    ROS_ERROR("Could not load parameter: 'orb_slam_frame'");
    ros::shutdown();
    return -1;
  }

  // Create SLAM system. It initializes all system threads and gets ready to
  // process frames.
  ORB_SLAM3::System SLAM(path_to_vocabulary,
                         path_to_settings,
                         ORB_SLAM3::System::STEREO,
                         viewer_on,
                         introspection_on);

  ResetServer rs(private_nh, &SLAM);
  OmnigraphInterface oi(private_nh, &SLAM);

  ImageGrabber igb(private_nh, orb_slam_frame, &SLAM);

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
      it, ros::names::remap("/stereo/left/image_raw"), 1, hints);
  image_transport::SubscriberFilter right_sub(
      it, ros::names::remap("/stereo/right/image_raw"), 1, hints);

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
  if (this->introspection_on) {
    mpSLAM->TrackStereoIntrospection(
        imLeft, imRight, cv_ptrLeft->header.stamp.toSec(), cost_img_cv);
  } else {
    mpSLAM->TrackStereo(imLeft, imRight, cv_ptrLeft->header.stamp.toSec());
  }

  // Publish the pose
  cv::Matx44f cv_pose;
  if (mpSLAM->GetCurrentCamPose(&cv_pose)) {
    geometry_msgs::Pose temp_pose = cvMatToPose(cv_pose);

    geometry_msgs::PoseStamped ros_pose_stamped;
    // Translation and rotation magic to escape camera frame
    ros_pose_stamped.pose.position.x = temp_pose.position.z;
    ros_pose_stamped.pose.position.y = -temp_pose.position.x;
    ros_pose_stamped.pose.position.z = -temp_pose.position.y;
    ros_pose_stamped.pose.orientation.x = temp_pose.orientation.x;
    ros_pose_stamped.pose.orientation.y = temp_pose.orientation.z;
    ros_pose_stamped.pose.orientation.z = -temp_pose.orientation.y;
    ros_pose_stamped.pose.orientation.w = temp_pose.orientation.w;

    // ros_pose_stamped.header.stamp =
    //    cv_ptrLeft->header.stamp;  // TODO This time may be old by now?
    ros_pose_stamped.header.stamp = ros::Time::now();
    ros_pose_stamped.header.frame_id = orb_slam_frame_;

    pose_pub_.publish(ros_pose_stamped);
  } else {
    ROS_WARN(
        "Could not retrieve pose to publish :( - maybe not fully initialized");
  }

  return;
}

bool ResetServer::ResetServerCB(std_srvs::Trigger::Request& req,
                                std_srvs::Trigger::Response& res) {
  if (mpSLAM_->GetTrackingState() == OK) {
    mpSLAM_->Reset();
    res.success = 1;
    res.message = "Called reset_vslam_server_";
    return true;
  } else {
    // Tracking is not ok - do not call reset becuase who knows what segfaults
    // we will cause :(
    res.success = 0;
    res.message =
        "Not able to call reset_vslam_server_ because tracking is not OK";
    return true;
  }
}

bool OmnigraphInterface::OmnigraphInterfaceCB(
    omnimapper_msgs::RelativePoseQuery::Request& req,
    omnimapper_msgs::RelativePoseQuery::Response& res) {
  cv::Matx44f cv_pose;
  if (mpSLAM_->GetCurrentCamPose(&cv_pose) &&
      mpSLAM_->GetTrackingState() == OK) {
    // Tracking is good and we got a pose - we can do what we need to do
    // Get relative transform
    cv::Matx44f cv_rel_pose =
        CalculateRelativeTransform(cv::Mat(this->last_pose), cv::Mat(cv_pose));
    // Update origin
    last_pose = cv_pose;
    geometry_msgs::Pose ros_rel_pose = cvMatToPose(cv_rel_pose);

    res.solution = req.initial_guess;
    res.solution.pose.position = ros_rel_pose.position;
    res.solution.pose.orientation = ros_rel_pose.orientation;

    // add covariance info
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) {
        if (r == c) {
          res.solution.covariance[6 * r + c] = 0.01;
        }
      }
    }
    res.success = 1;
    ROS_WARN("Good visual odom");
    return true;
  } else {
    // Tracking is not ok - we cannot do what we need to do
    res.success = 0;
    ROS_WARN("Failed visual odom");
    return true;
  }
}

cv::Mat CalculateInverseTransform(const cv::Mat& transform) {
  if (transform.empty()) {
    std::cout << "MATRIX IS EMPTY!! " << std::endl;
  }

  cv::Mat R1 = transform.rowRange(0, 3).colRange(0, 3);
  cv::Mat t1 = transform.rowRange(0, 3).col(3);
  cv::Mat R1_inv = R1.t();
  cv::Mat t1_inv = -R1_inv * t1;
  cv::Mat transform_inv = cv::Mat::eye(4, 4, transform.type());

  R1_inv.copyTo(transform_inv.rowRange(0, 3).colRange(0, 3));
  t1_inv.copyTo(transform_inv.rowRange(0, 3).col(3));

  return transform_inv;
}

cv::Mat CalculateRelativeTransform(const cv::Mat& dest_frame_pose,
                                   const cv::Mat& src_frame_pose) {
  return CalculateInverseTransform(dest_frame_pose) * src_frame_pose;
}

/*
bool service_callback() {
  ROS_DEBUG_STREAM(fixed << "[VIWO] Service callback");
  if (!have_base_calibration) {
    if (get_imutobase_calibration())
      have_base_calibration = true;
    else {
      ROS_DEBUG("[VIWO] Check base calibration failed");
      return false;
    }
  }

  // get current state
  State* state = sys_->get_state();
  if (!sys_->initialized()) return false;

  graph_node_info prev_node, new_node;
  set_graph_nodes(req, prev_node, new_node);

  update_keyframes(state, prev_node, new_node);

  resp.success = calculate_odometry(state, prev_node, new_node, resp.solution);

  ROS_INFO_STREAM(
      "VIWO REQ INIT GUESS FROM WHEEL ODOM: " << req.initial_guess.pose);
  ROS_INFO_STREAM("VIWO RESP RETURNING : "
                  << resp.solution.pose << " Success ? " << (int)resp.success);
  ROS_INFO_STREAM(fixed << "[VIWO] CALLBACK SVC::prev_node: " << prev_node.time
                        << " new node:" << new_node.time);
  return true;
}
*/
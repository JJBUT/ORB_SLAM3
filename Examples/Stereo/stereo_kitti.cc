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

#include <System.h>
#include <gflags/gflags.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>

#include "iv_slam_helpers/torch_helpers.h"

DECLARE_bool(help);

DEFINE_string(path_to_vocabulary, "", "Absolute path to the ORB vocabulary.");
DEFINE_string(path_to_settings, "", "Absolute path to the settings.");
DEFINE_string(path_to_sequences,
              "",
              "Absolute path to the stereo image sequences.");
DEFINE_string(path_to_output_training_data,
              "",
              "Path to write the training data to.");
DEFINE_string(path_to_introspection_model,
              "",
              "Absolute path to the introspection model");

DEFINE_bool(introspection_on,
            false,
            "Run ORB-SLAM3 with the introspection function - GPU suggested.");
DEFINE_bool(generate_training_data_on,
            false,
            "Given an image sequence and ground truth posese generate training "
            "data for the intropsection model ");
DEFINE_bool(
    visualize_groundtruth_on,
    false,
    "Visualize the ground truth keyframe and camera poses - relative to some "
    "buffer - not in an absolute sense because drift makes it meaningless");
DEFINE_bool(gpu_available, false, "Set to true if a GPU is available to use.");
DEFINE_bool(viewer_on, true, "Enable image and keyframe viewer.");
DEFINE_bool(undistort_rectify_on,
            false,
            "Undistort and/or rectify images. If this is set to true both the "
            "distortion and rectification calibration must be set. If you set "
            "the parameters to zero then it will not perfrom that respective "
            "correction (i.e. undistort or rectify)");

using namespace std;

void LoadImages(const string &strPathToSequence,
                const bool generate_training_data_on,
                vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight,
                vector<double> &vTimestamps,
                vector<cv::Mat> &vGroundTruthPoses);

ORB_SLAM3::System *SLAM_ptr;

void SignalHandler(int signal_num) {
  cout << "Interrupt signal is (" << signal_num << ").\n";

  // terminate program
  if (SLAM_ptr) {
    SLAM_ptr->Shutdown();
  }

  cout << "Exiting the program!" << endl;

  exit(signal_num);
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  signal(SIGINT, SignalHandler);

  // Load introspection model
  torch::jit::script::Module introspection_model;
  torch::Device device = torch::kCPU;
  if (FLAGS_introspection_on) {
    // Check if we have a GPU to run on
    if (FLAGS_gpu_available && torch::cuda::is_available()) {
      device = torch::kCUDA;
      cout << "Introspection model running on GPU :)" << endl;
    } else {
      cout << "Introspection model running on CPU :(" << endl;
    }
    try {
      // Deserialize the ScriptModule from file
      introspection_model = torch::jit::load(FLAGS_path_to_introspection_model);
      introspection_model.to(device);
    } catch (const c10::Error &e) {
      cerr << "Error deserializing the ScriptModule from file" << endl;
      return -1;
    }
  }

  // Generate undistortion and/or rectification maps if requested
  cv::Mat M1l, M2l, M1r, M2r;
  if (FLAGS_undistort_rectify_on) {
    // Load settings related to stereo calibration
    cv::FileStorage fsSettings(FLAGS_path_to_settings, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
      cerr << "ERROR: Wrong path to settings" << endl;
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
      cerr << "ERROR: Calibration parameters to rectify stereo are missing!"
           << endl;
      return -1;
    }

    cv::initUndistortRectifyMap(K_l,
                                D_l,
                                R_l,
                                P_l.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_l, rows_l),
                                CV_32F,
                                M1l,
                                M2l);
    cv::initUndistortRectifyMap(K_r,
                                D_r,
                                R_r,
                                P_r.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_r, rows_r),
                                CV_32F,
                                M1r,
                                M2r);
  }

  // Retrieve paths to images
  vector<string> vstrImageLeft;
  vector<string> vstrImageRight;
  vector<double> vTimestamps;
  vector<cv::Mat> vGroundTruthPoses;  // Ground truth poses are only loaded if
                                      // the generate training data flag is true
  LoadImages(string(FLAGS_path_to_sequences),
             bool(FLAGS_generate_training_data_on),
             vstrImageLeft,
             vstrImageRight,
             vTimestamps,
             vGroundTruthPoses);

  const int nImages = vstrImageLeft.size();

  // Create SLAM system. It initializes all system threads and gets ready to
  // process frames.
  SLAM_ptr = new ORB_SLAM3::System(FLAGS_path_to_vocabulary,
                                   FLAGS_path_to_settings,
                                   ORB_SLAM3::System::STEREO,
                                   FLAGS_viewer_on,
                                   FLAGS_introspection_on,
                                   FLAGS_generate_training_data_on,
                                   FLAGS_visualize_groundtruth_on);

  // Vector for tracking time statistics
  vector<float> vTimesTrack;
  vTimesTrack.resize(nImages);

  cout << endl << "-------" << endl;
  cout << "Start processing sequence ..." << endl;
  cout << "Images in the sequence: " << nImages << endl << endl;

  // Main loop
  cv::Mat imLeftOriginal, imRightOriginal;
  for (int ni = 0; ni < nImages; ni++) {
    // Read left and right images from file
    imLeftOriginal = cv::imread(vstrImageLeft[ni], cv::IMREAD_UNCHANGED);
    imRightOriginal = cv::imread(vstrImageRight[ni], cv::IMREAD_UNCHANGED);
    double tframe = vTimestamps[ni];

    if (imLeftOriginal.empty()) {
      cerr << endl
           << "Failed to load image at: " << string(vstrImageLeft[ni]) << endl;
      return 1;
    }

    cv::Mat imLeft, imRight;
    if (FLAGS_undistort_rectify_on) {
      cv::remap(imLeftOriginal, imLeft, M1l, M2l, cv::INTER_LINEAR);
      cv::remap(imRightOriginal, imRight, M1r, M2r, cv::INTER_LINEAR);
    } else {
      // Don't undistort/rectify
      imLeft = imLeftOriginal;
      imRight = imRightOriginal;
    }

    // Feed image to model to create cost mask
    cv::Mat cost_image_cv;
    at::Tensor cost_img;
    if (FLAGS_introspection_on) {
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

      tensor_img = tensor_img.to(device);
      std::vector<torch::jit::IValue> inputs{tensor_img};
      cost_img = introspection_model.forward(inputs).toTensor();
      cost_img = (cost_img * 255.0).to(torch::kByte);
      cost_img = cost_img.to(torch::kCPU);

      cost_image_cv = ORB_SLAM3::ToCvImage(cost_img);
    }

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t1 =
        std::chrono::monotonic_clock::now();
#endif

    // Pass the images to the SLAM system
    if (SLAM_ptr->IntrospectionOn()) {
      SLAM_ptr->TrackStereoIntrospection(
          imLeft, imRight, tframe, cost_image_cv);
    } else if (SLAM_ptr->GenerateTrainingDataOn()) {
      SLAM_ptr->TrackStereoTrainingDataGeneration(
          imLeft, imRight, tframe, vGroundTruthPoses[ni]);
    } else {
      SLAM_ptr->TrackStereo(imLeft, imRight, tframe);
    }

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t2 =
        std::chrono::monotonic_clock::now();
#endif

    double ttrack =
        std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1)
            .count();

    vTimesTrack[ni] = ttrack;

    // Wait to load the next frame
    double T = 0;
    if (ni < nImages - 1)
      T = vTimestamps[ni + 1] - tframe;
    else if (ni > 0)
      T = tframe - vTimestamps[ni - 1];

    if (ttrack < T) usleep((T - ttrack) * 1e6);
  }

  // Stop all threads
  SLAM_ptr->Shutdown();

  // Tracking time statistics
  sort(vTimesTrack.begin(), vTimesTrack.end());
  float totaltime = 0;
  for (int ni = 0; ni < nImages; ni++) {
    totaltime += vTimesTrack[ni];
  }
  cout << "-------" << endl << endl;
  cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
  cout << "mean tracking time: " << totaltime / nImages << endl;

  // Save camera trajectory
  SLAM_ptr->SaveTrajectoryKITTI("CameraTrajectory.txt");

  return 0;
}

void LoadImages(const string &strPathToSequence,
                const bool generate_training_data_on,
                vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight,
                vector<double> &vTimestamps,
                vector<cv::Mat> &vGroundTruthPoses) {
  ifstream fTimes;
  string strPathTimeFile = strPathToSequence + "/times.txt";
  fTimes.open(strPathTimeFile.c_str());
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      double t;
      ss >> t;
      vTimestamps.push_back(t);
    }
  }

  string strPrefixLeft = strPathToSequence + "/image_0/";
  string strPrefixRight = strPathToSequence + "/image_1/";

  const int nTimes = vTimestamps.size();
  vstrImageLeft.resize(nTimes);
  vstrImageRight.resize(nTimes);

  for (int i = 0; i < nTimes; i++) {
    stringstream ss;
    ss << setfill('0') << setw(6) << i;
    vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
    vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
  }

  // Load ground truth relative poses
  if (generate_training_data_on) {
    ifstream fGroundTruthPoses;
    string strGroundTruthPosesFile = strPathToSequence + "/poses.txt";
    fGroundTruthPoses.open(strGroundTruthPosesFile.c_str());
    while (!fGroundTruthPoses.eof()) {
      string s;
      getline(fGroundTruthPoses, s);
      if (!s.empty()) {
        cv::Mat camera_pose = cv::Mat::eye(4, 4, CV_32F);
        stringstream ss(s);
        string str_current_entry;

        for (size_t i = 0; i < 12; i++) {
          getline(ss, str_current_entry, ' ');
          camera_pose.at<float>(floor((float)(i) / 4), i % 4) =
              stof(str_current_entry);
        }
        vGroundTruthPoses.push_back(camera_pose);
      }
    }
    CHECK(nTimes == vGroundTruthPoses.size())
        << ": Each timestamp/image set does not have a matching pose :(";
  }
}
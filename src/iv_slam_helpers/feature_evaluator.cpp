// Copyright 2019 srabiee@cs.umass.edu
// College of Information and Computer Sciences,
// University of Massachusetts Amherst
//
//
// This software is free: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License Version 3,
// as published by the Free Software Foundation.
//
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// Version 3 in the file COPYING that came with this distribution.
// If not, see <http://www.gnu.org/licenses/>.
// ========================================================================

#include "feature_evaluator.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>

#include "Frame.h"

namespace feature_evaluation {

using Eigen::MatrixXf;
using Eigen::Vector2f;
using Eigen::VectorXf;

void GenerateUnsupImageQualityHeatmapGP(ORB_SLAM3::Frame& frame,
                                        const cv::Mat img_curr,
                                        cv::Mat* bad_region_heatmap_ptr,
                                        cv::Mat* bad_region_heatmap_mask_ptr) {
  if (!bad_region_heatmap_ptr || !bad_region_heatmap_mask_ptr) {
    LOG(ERROR) << "Bad ptr -_-";
    return;
  }
  cv::Mat& bad_region_heatmap = *bad_region_heatmap_ptr;
  cv::Mat& bad_region_heatmap_mask = *bad_region_heatmap_mask_ptr;

  // The threshold that is used to generate a binary mask from the
  // Gaussian Process estimated variance values that are normalized btw 0 and 1
  const float kNormalizedGPVarThresh = 0.5;

  // The maximum threshold used for normalizing the estimated variance values
  // by the Gaussian Process.
  const float kGPVarMaxThresh = 100.0;  // 200.0

  int const N = frame.N;
  vector<int> idx_interest;
  idx_interest.reserve(N);

  for (size_t i = 0; i < N; i++) {
    if (frame.mvChi2Dof[i] > 0) {
      idx_interest.push_back(i);
    }
  }

  vector<float> err_vals_vec(idx_interest.size());
  vector<Vector2f> point_loc(idx_interest.size());

  for (size_t i = 0; i < idx_interest.size(); i++) {
    err_vals_vec[i] =
        (2 / (1 + frame.mvKeyQualScoreTrain[idx_interest[i]])) - 1;
    point_loc[i] = Vector2f(frame.mvKeysUn[idx_interest[i]].pt.x,
                            frame.mvKeysUn[idx_interest[i]].pt.y);
  }

  // Image quality heatmap parameters
  const float kBinSizeX = 40.0;  // 200, *100, 50, +40
  const float kBinSizeY = 40.0;
  const float kBinStride = 20.0;  // +20

  // The remaining strip at the right and bottom of the image are cropped out
  // This should be taken into accout during training
  int bin_num_x =
      std::floor((double(img_curr.cols) - kBinSizeX) / kBinStride) + 1;
  int bin_num_y =
      std::floor((double(img_curr.rows) - kBinSizeY) / kBinStride) + 1;

  if (err_vals_vec.empty()) {
    LOG(INFO) << "No keypoints available for heatmap generation!";
    bad_region_heatmap = cv::Mat((bin_num_y - 1) * kBinStride + kBinSizeY,
                                 (bin_num_x - 1) * kBinStride + kBinSizeX,
                                 CV_8U);
    return;
  }

  MatrixXf kmat = Kmatrix(point_loc);
  Eigen::Map<VectorXf> err_vals(err_vals_vec.data(), err_vals_vec.size());

  vector<double> grid_quality_vec(bin_num_x * bin_num_y);
  vector<double> grid_qual_var_vec(bin_num_x * bin_num_y);

  // Predict/interpolate the error value for each of the points on a grid
  for (int j = 0; j < bin_num_y; j++) {
    for (int i = 0; i < bin_num_x; i++) {
      float x = i * kBinStride + kBinSizeX / 2.0;
      float y = j * kBinStride + kBinSizeY / 2.0;
      float mean;
      float variance;
      GPPredict(x, y, point_loc, err_vals, kmat, mean, variance);
      grid_quality_vec[i + j * bin_num_x] = static_cast<double>(mean);
      grid_qual_var_vec[i + j * bin_num_x] = static_cast<double>(variance);
    }
  }

  // +++++++++++++++
  // Use the GP variance values to generate a reliability mask for the
  // image quality heatmap
  cv::Mat bad_region_var_heatmap_low_res = GenerateErrHeatmap(
      bin_num_y, bin_num_x, grid_qual_var_vec, kGPVarMaxThresh, 0.0);

  // Scale up the heatmap to the original image size (minus the cropped out
  // stripes) and convert it to CV_8U
  cv::resize(bad_region_var_heatmap_low_res,
             bad_region_heatmap_mask,
             cv::Size((bin_num_x - 1) * kBinStride + kBinSizeX,
                      (bin_num_y - 1) * kBinStride + kBinSizeY));

  //   cv::imshow("GP variance heatmap", bad_region_heatmap_mask);

  cv::threshold(bad_region_heatmap_mask,
                bad_region_heatmap_mask,
                kNormalizedGPVarThresh,
                1.0,
                cv::THRESH_BINARY_INV);

  bad_region_heatmap_mask.convertTo(bad_region_heatmap_mask, CV_8U, 255.0);

  //   cv::imshow("GP heatmap mask", bad_region_heatmap_mask);
  // --------------

  // Create a heatmap image of the bad regions for SLAM/VO
  cv::Mat bad_region_heatmap_low_res =
      GenerateErrHeatmap(bin_num_y, bin_num_x, grid_quality_vec, 1.0, 0.0);

  // Scale up the heatmap to the original image size (minus the cropped out
  // stripes) and convert it to CV_8U
  cv::resize(bad_region_heatmap_low_res,
             bad_region_heatmap,
             cv::Size((bin_num_x - 1) * kBinStride + kBinSizeX,
                      (bin_num_y - 1) * kBinStride + kBinSizeY));

  bad_region_heatmap.convertTo(bad_region_heatmap, CV_8U, 255.0);

  return;
}

inline float Kernel(const Vector2f& x1, const Vector2f& x2) {
  const float s_f = 80.0;
  const float l = 100.0;  // def: 200

  return s_f * s_f * exp(-1.0 / (2.0 * l * l) * ((x1 - x2).squaredNorm()));
}

MatrixXf Kmatrix(const vector<Vector2f>& X) {
  //   const float s_n = 2.0;
  const float s_n = 20.0;

  int N = X.size();
  MatrixXf Km(N, N), I(N, N);
  I.setIdentity();
  I.setIdentity();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      Km(i, j) = Kernel(X[i], X[j]);
      Km(j, i) = Km(i, j);
    }
  }
  Km = (Km + s_n * s_n * I).inverse();
  return Km;
}

void GPPredict(float x,
               float y,
               const vector<Vector2f>& locs,
               const VectorXf& values,
               const MatrixXf& K_mat,
               float& mean,
               float& variance) {
  const double kErrMinClamp = 0.0;
  const float kErrMean = kErrMinClamp;

  int N = locs.size();
  MatrixXf Kv(N, 1), Kvt;
  Vector2f l(x, y);
  for (int i = 0; i < N; i++) {
    Kv(i, 0) = Kernel(l, locs[i]);
  }
  Kvt = Kv.transpose();
  MatrixXf u = Kvt * K_mat;
  MatrixXf ret;
  ret = u * values;
  mean = ret(0, 0) + kErrMean;
  ret = (u * Kv);
  variance = ret(0, 0);
  variance = Kernel(l, l) - variance;
}

cv::Mat GenerateErrHeatmap(unsigned int rows,
                           unsigned int cols,
                           const std::vector<double> err_vals,
                           double err_max_clamp,
                           double err_min_clamp) {
  double pv[err_vals.size()];
  for (unsigned int i = 0; i < err_vals.size(); i++) {
    double scaled_err = static_cast<double>((err_vals[i] - err_min_clamp) /
                                            (err_max_clamp - err_min_clamp));
    scaled_err = (scaled_err > 1.0) ? 1.0 : scaled_err;
    scaled_err = (scaled_err < 0.0) ? 0.0 : scaled_err;
    pv[i] = scaled_err;
  }

  cv::Mat out_img(rows, cols, CV_64FC1);
  memcpy(out_img.data, &pv, err_vals.size() * sizeof(err_vals[0]));

  return out_img;
}

bool IsHeatmapMaskAllZero(const cv::Mat& bad_region_heatmap_mask) {
  if (bad_region_heatmap_mask.empty()) {
    return true;
  } else {
    if (cv::sum(bad_region_heatmap_mask)[0] > 0) {
      return false;
    } else {
      return true;
    }
  }
}

}  // namespace feature_evaluation

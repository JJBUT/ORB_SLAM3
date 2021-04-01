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
#ifndef IVSLAM_FEATURE_EVALUATOR
#define IVSLAM_FEATURE_EVALUATOR

#include <Eigen/Core>
#include <vector>

#include "Frame.h"

namespace feature_evaluation {

void GenerateUnsupImageQualityHeatmapGP(ORB_SLAM3::Frame& frame,
                                        const cv::Mat img_curr,
                                        cv::Mat* bad_region_heatmap_ptr,
                                        cv::Mat* bad_region_heatmap_mask_ptr);

Eigen::MatrixXf Kmatrix(const std::vector<Eigen::Vector2f>& X);

// Gaussian process prediction
void GPPredict(float x,
               float y,
               const std::vector<Eigen::Vector2f>& locs,
               const Eigen::VectorXf& values,
               const Eigen::MatrixXf& K_mat,
               float& mean,
               float& variance);

cv::Mat GenerateErrHeatmap(unsigned int rows,
                           unsigned int cols,
                           const std::vector<double> err_vals,
                           double err_max_clamp,
                           double err_min_clamp);
}  // namespace feature_evaluation

#endif  // IVSLAM_FEATURE_EVALUATOR

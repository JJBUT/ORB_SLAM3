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
#ifndef iSLAM_DATASET_CREATOR
#define iSLAM_DATASET_CREATOR

#include <jsoncpp/json/json.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace feature_evaluation {

class DatasetCreator {
 public:
  DatasetCreator(const std::string& dataset_path);

  ~DatasetCreator() = default;

  void SaveToFile();

  void SaveBadRegionHeatmap(const std::string& img_name,
                            const cv::Mat& bad_region_heatmap);

  void SaveBadRegionHeatmapMask(const std::string& img_name,
                                const cv::Mat& bad_region_heatmap_mask);

 private:
  const std::string img_names_file_name_ = "img_names.json";

  std::string dataset_path_;
  Json::Value img_names_json_;
};

}  // namespace feature_evaluation

#endif  // iSLAM_DATASET_CREATOR

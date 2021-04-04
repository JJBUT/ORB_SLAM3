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

#include "dataset_creator.h"

#include <dirent.h>
#include <glog/logging.h>
#include <sys/stat.h>

#include "io_access.h"
// #include <bitset>

namespace feature_evaluation {

using std::string;

using namespace ORB_SLAM3;

DatasetCreator::DatasetCreator(const string& dataset_path)
    : dataset_path_(dataset_path) {
  if (!CreateDirectory(dataset_path_)) {
    LOG(FATAL) << "Could not create the directory for saving the descriptors";
  }
}

void DatasetCreator::SaveToFile() {
  // Write the heatmap/image names to file.
  if (!WriteJsonToFile(
          dataset_path_, "/" + img_names_file_name_, img_names_json_)) {
    LOG(FATAL) << "Could not create directory "
               << dataset_path_ + "/" + img_names_file_name_;
  }

  return;
}

void DatasetCreator::SaveBadRegionHeatmap(const string& img_name,
                                          const cv::Mat& bad_region_heatmap) {
  string img_dir = dataset_path_ + "/bad_region_heatmap/";
  if (!CreateDirectory(img_dir)) {
    LOG(FATAL) << "Could not create the directory for saving the heatmaps";
  }

  string img_path = img_dir + img_name;
  cv::imwrite(img_path, bad_region_heatmap);

  img_names_json_["img_name"].append(img_name);

  return;
}

void DatasetCreator::SaveBadRegionHeatmapMask(
    const string& img_name, const cv::Mat& bad_region_heatmap_mask) {
  string img_dir = dataset_path_ + "/bad_region_heatmap_mask/";
  if (!CreateDirectory(img_dir)) {
    LOG(FATAL) << "Could not create the directory for saving the heatmaps";
  }

  string img_path = img_dir + img_name;
  cv::imwrite(img_path, bad_region_heatmap_mask);

  return;
}

}  // namespace feature_evaluation

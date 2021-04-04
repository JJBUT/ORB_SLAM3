#!/bin/bash

# General required settings/paths
PATH_TO_READ_VOCABULARY="/home/administrator/SLAM/ORB_SLAM3/Vocabulary/ORBvoc.txt"
PATH_TO_READ_SETTINGS="/home/administrator/SLAM/ORB_SLAM3/Examples/Stereo/ahg_turn_test.yaml"
PATH_TO_READ_SEQUENCES="/home/administrator/DATA/KITTI_FORMAT/00015/sequences"

# Required settings/paths for introspection
INTROSPECTION_ON="true"
GPU_AVAILABLE="true"
PATH_TO_READ_INTROSPECTION_MODEL="/home/administrator/DATA/MODEL/speedway_24th_cross/exported_model/iv_speedway_24th_cross_mobilenet_c1deepsup_light.pt"

# Required settings/paths for training data generation
GENERATE_TRAINING_DATA_ON="false"
PATH_TO_WRITE_OUTPUT_TRAINING_DATA="/home/administrator/Desktop"

# General settings
UNDISTORT_RECTIFY_ON="true" # If your images are not already undistorted 
                            # (or recitifed for stereo) you must set this 
                            # to true and provided the calibration in the 
                            # settings yaml

# Settings for the viewer 
VIEWER_ON="true"
VISUALIZE_GROUNTRUTH_ON="false" # Visualise groundtruth can only be requested 
                                # when you are doing training data generation 
                                # because you provide the groundtruth pose


../../Examples/Stereo/stereo_kitti \
    --path_to_vocabulary=$PATH_TO_READ_VOCABULARY \
    --path_to_settings=$PATH_TO_READ_SETTINGS \
    --path_to_sequences=$PATH_TO_READ_SEQUENCES \
    --path_to_output_training_data=$PATH_TO_WRITE_OUTPUT_TRAINING_DATA \
    --path_to_introspection_model=$PATH_TO_READ_INTROSPECTION_MODEL \
    --introspection_on=$INTROSPECTION_ON \
    --generate_training_data_on=$GENERATE_TRAINING_DATA_ON \
    --gpu_available=$GPU_AVAILABLE \
    --undistort_rectify_on=$UNDISTORT_RECTIFY_ON \
    --viewer_on=$VIEWER_ON \
    --visualize_groundtruth_on=$VISUALIZE_GROUNTRUTH_ON 
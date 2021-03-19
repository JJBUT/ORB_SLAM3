#!/bin/bash
PATH_TO_VOCABULARY="/home/administrator/SLAM/ORB_SLAM3/Vocabulary/ORBvoc.txt"
PATH_TO_SETTINGS="/home/administrator/SLAM/ORB_SLAM3/Examples/Stereo/ahg_turn_test.yaml"
PATH_TO_SEQUENCES="/home/administrator/DATA/KITTI_FORMAT/00015/sequences"
PATH_TO_INTROSPECTION_MODEL="/home/administrator/DATA/MODEL/speedway_24th_cross/exported_model/iv_speedway_24th_cross_mobilenet_c1deepsup_light.pt"

INTROSPECTION_ON="true"
VIEWER_ON="true"
GPU_AVAILABLE="true"
# TODO undistort and rectify

../../Examples/Stereo/stereo_kitti --path_to_vocabulary=$PATH_TO_VOCABULARY --path_to_settings=$PATH_TO_SETTINGS --path_to_sequences=$PATH_TO_SEQUENCES --path_to_introspection_model=$PATH_TO_INTROSPECTION_MODEL --introspection_on=$INTROSPECTION_ON --viewer_on=$VIEWER_ON --gpu_available=$GPU_AVAILABLE

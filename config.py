#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# config.py
#  Configuration dictionary for VoxelNet project
#  Created by G. Dylan Dickerson on 10 Dec 2019

import os
from easydict import EasyDict as edict
import math

possibleOBJ = ('Car', 'Pedestrian', 'Cyclist')
cfg = edict()

# Helper functions
def selectTargetVoxels(config_t):
    if config_t.TARGET_OBJ not in possibleOBJ:
        raise Exception("Invalid target object type selected")
    
    if config_t.TARGET_OBJ is 'Car':  # Set the image boundaries in meters
        config_t.X_MIN = 0
        config_t.X_MAX = 70.4
        config_t.Y_MIN = -40
        config_t.Y_MAX = 40
        config_t.Z_MIN = -3
        config_t.Z_MAX = 1
        config_t.POINT_PER_VOX = 35   # The max number of voxels randomly sampled from each voxel

    else:                 # Is a Pedestrian or Cyclist
        config_t.X_MIN = 0
        config_t.X_MAX = 48
        config_t.Y_MIN = -20
        config_t.Y_MAX = 20
        config_t.Z_MIN = -3
        config_t.Z_MAX = 1
        config_t.POINT_PER_VOX = 45
    # Calculate number of voxels in input field
    config_t.INPUT_WIDTH = int((config_t.X_MAX - config_t.X_MIN) / (config.VOXEL_WIDTH))
    config_t.INPUT_HEIGHT = int((config_t.Y_MAX - config_t.Y_MIN) / (config.VOXEL_HEIGHT))
    config_t.INPUT_DEPTH = int((config_t.Z_MAX - config_t.Z_MIN) / (config.VOXEL_DEPTH))

    config_t.FEATURE_WIDTH = int(config_t.INPUT_WIDTH / config_t.FEATURE_RATIO)
    config_t.FEATURE_HEIGHT = int(config_t.INPUT_HEIGHT / config_t.FEATURE_RATIO)
    return


def selectTargetAnchors(config_t):
    if config_t.TARGET_OBJ not in possibleOBJ:
        raise Exception("Invalid target object type selected")

    if config_t.TARGET_OBJ is 'Car':
        # Anchor dimensions in meters
        config_t.ANCHOR_LEN = 3.9
        config_t.ANCHOR_WID = 1.6
        config_t.ANCHOR_HEI = 1.56
        # Anchor Z center
        config_t.ANCHOR_Z = -1.0 - config_t.ANCHOR_HEI / 2
        config_t.RPN_POS_IOU = 0.6      # Anchor Positive IoU minimum (positive match)
        config_t.RPN_NEG_IOU = 0.45     # Anchor Negative IoU max (non-match)

    elif config_t.TARGET_OBJ is 'Pedestrian':
        # Anchor dimensions in meters
        config_t.ANCHOR_LEN = 0.8
        config_t.ANCHOR_WID = 0.6
        config_t.ANCHOR_HEI = 1.73
        # Anchor Z center
        config_t.ANCHOR_Z = -0.6 - config_t.ANCHOR_HEI / 2
        config_t.RPN_POS_IOU = 0.5      # Anchor Positive IoU minimum (positive match)
        config_t.RPN_NEG_IOU = 0.35     # Anchor Negative IoU max (non-match)

    elif config_t.TARGET_OBJ is 'Cyclist':
        # Anchor dimensions in meters
        config_t.ANCHOR_LEN = 1.76
        config_t.ANCHOR_WID = 0.6
        config_t.ANCHOR_HEI = 1.73
        # Anchor Z center
        config_t.ANCHOR_Z = -0.6 - config_t.ANCHOR_HEI / 2
        config_t.RPN_POS_IOU = 0.5      # Anchor Positive IoU minimum (positive match)
        config_t.RPN_NEG_IOU = 0.35     # Anchor Negative IoU max (non-match)
    return


def changeTargetObj(target, config_t):
    if target not in possibleOBJ:
        raise Exception("Invalid target object type selected")

    config_t.TARGET_OBJ = target
    selectTargetVoxels(config_t)
    selectTargetAnchors(config_t)
    return


# Directory Info
cfg.ROOT_DIR = os.getcwd()
cfg.CHECKPOINT_DIR = os.path.join(cfg.ROOT_DIR, 'checkpoint')
cfg.LOG_DIR = os.path.join(cfg.ROOT_DIR, 'logs')
cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, 'data')

# GPU Info
cfg.GPU_AVAILABLE = '0'
cfg.GPU_USE_COUNT = len(cfg.GPU_AVAILABLE.split(','))

# General
cfg.BV_LOG_FACTOR = 8       # Image log scale factor
cfg.FEATURE_RATIO = 2
cfg.USE_CORNER2CENTER_AVG = True    # Use average or max version
## For Region Proposal Network's Non-Maximal Sort
cfg.RPN_NMS_POST_TOPK = 20
cfg.RPN_NMS_THRESH = 0.3
cfg.RPN_SCORE_THRESH = 0.96
## For 2d proposal to 3d proposal
cfg.PROPOSAL3D_Z_MIN = -2.3
cfg.PROPOSAL3D_Z_MAX = 1.5

# Target and Voxel info
cfg.VOXEL_WIDTH = 0.2       # Voxel width in meters
cfg.VOXEL_HEIGHT = 0.2      # Voxel height in meters
cfg.VOXEL_DEPTH = 0.4       # Voxel depth in meters
changeTargetObj('Car', cfg)

# Dataset Info
cfg.DATA_SETS_TYPE = 'kitti'

# Sensor info
cfg.VELODYNE_ANGULAR_RESOLUTION = 0.08 / 180 * math.pi
cfg.VELODYNE_VERTICAL_RESOLUTION = 0.4 / 180 * math.pi
cfg.VELODYNE_HEIGHT = 1.73
## RBG
if cfg.DATA_SETS_TYPE == 'kitti':
    cfg.IMAGE_WIDTH = 1242
    cfg.IMAGE_HEIGHT = 375
    cfg.IMAGE_CHANNEL = 3
# Top
if cfg.DATA_SETS_TYPE == 'kitti':
    cfg.TOP_Y_MIN = -30
    cfg.TOP_Y_MAX = +30
    cfg.TOP_X_MIN = 0
    cfg.TOP_X_MAX = 80
    cfg.TOP_Z_MIN = -4.2
    cfg.TOP_Z_MAX = 0.8

    cfg.TOP_X_DIVISION = 0.1
    cfg.TOP_Y_DIVISION = 0.1
    cfg.TOP_Z_DIVISION = 0.2

    cfg.TOP_WIDTH = (cfg.TOP_X_MAX - cfg.TOP_X_MIN) // cfg.TOP_X_DIVISION
    cfg.TOP_HEIGHT = (cfg.TOP_Y_MAX - cfg.TOP_Y_MIN) // cfg.TOP_Y_DIVISION
    cfg.TOP_CHANNEL = (cfg.TOP_Z_MAX - cfg.TOP_Z_MIN) // cfg.TOP_Z_DIVISION

# For camera and lidar coordination convert
if cfg.DATA_SETS_TYPE == 'kitti':
    # cal mean from train set
    cfg.MATRIX_P2 = ([[719.787081,    0.,            608.463003, 44.9538775],
                      [0.,            719.787081,    174.545111, 0.1066855],
                      [0.,            0.,            1.,         3.0106472e-03],
                      [0.,            0.,            0.,         0]])

    # cal mean from train set
    cfg.MATRIX_T_VELO_2_CAM = ([
        [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
        [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
        [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
        [0, 0, 0, 1]
    ])
    # cal mean from train set
    cfg.MATRIX_R_RECT_0 = ([
        [0.99992475, 0.00975976, -0.00734152, 0],
        [-0.0097913, 0.99994262, -0.00430371, 0],
        [0.00729911, 0.0043753, 0.99996319, 0],
        [0, 0, 0, 1]
    ])




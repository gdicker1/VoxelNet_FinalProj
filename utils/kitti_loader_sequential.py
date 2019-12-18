#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import cv2
import numpy as np
import os
import sys
import glob
import time
import math
import random
from sklearn.utils import shuffle

from config import cfg
from utils.data_aug import aug_data
from utils.preprocess import process_pointcloud

class KittiLoaderSeq(object):
	# return:
	# tag (N)
	# label (N) (N')
	# rgb (N, H, W, C)
	# raw_lidar (N) (N', 4)
	# vox_feature
	# vox_number
	# vox_coordinate

	def __init__(self, object_dir='.', require_shuffle=False, is_testset=True, batch_size=1, split_file='', aug=False, aug_num=0):
		self.object_dir = object_dir
		self.is_testset = is_testset
		self.require_shuffle = require_shuffle if not self.is_testset else False
		self.batch_size = batch_size
		self.split_file = split_file
		self.aug = aug
		self.aug_num = aug_num

		if self.split_file != '':
			# use split file to find file names
			_tag = []
			self.f_rgb, self.f_lidar, self.f_label = [], [], []
			for line in open(self.split_file, 'r').readlines():
				line = line[:-1]  # remove '\n'
				_tag.append(line)
				self.f_rgb.append(os.path.join(
					self.object_dir, 'image_2', line + '.png'))
				self.f_lidar.append(os.path.join(
					self.object_dir, 'velodyne', line + '.bin'))
				self.f_label.append(os.path.join(
					self.object_dir, 'label_2', line + '.txt'))
		else:
			# get all file names from object_dir
			self.f_rgb = glob.glob(os.path.join(
				self.object_dir, 'image_2', '*.png'))
			self.f_rgb.sort()
			self.f_lidar = glob.glob(os.path.join(
				self.object_dir, 'velodyne', '*.bin'))
			self.f_lidar.sort()
			self.f_label = glob.glob(os.path.join(
				self.object_dir, 'label_2', '*.txt'))
			self.f_label.sort()

		self.data_tag = [name.split(os.path.sep)[-1].split('.')[-2]
						 for name in self.f_rgb]
		assert(len(self.data_tag) == len(self.f_rgb) == len(self.f_lidar))

		self.dataset_size = len(self.f_rgb)
		print("Dataset total length: {}".format(self.dataset_size))
		self.rgb_shape = (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3)
		self.load_index = 0

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		return

	def __len__(self):
		return self.dataset_size

	def reset(self):
		self.load_index = 0
		if self.require_shuffle:
			self.shuffle_dataset()


	def shuffle_dataset(self):
		# to prevent diff loader load same data
		index = shuffle([i for i in range(len(self.f_label))],
						random_state=random.randint(0, 8**5))
		self.f_label = [self.f_label[i] for i in index]
		self.f_rgb = [self.f_rgb[i] for i in index]
		self.f_lidar = [self.f_lidar[i] for i in index]
		self.data_tag = [self.data_tag[i] for i in index]


	def load(self, batch_size):
		if self.load_index >= self.dataset_size:
			return None

		num_to_load = batch_size
		
		labels, tag, voxel, rgb, raw_lidar = [], [], [], [], []
		vox_feature, vox_number, vox_coordinate = [], [], []
		# Load normal data
		for i in range(batch_size):
			load_index = self.load_index + i
			rgb.append(cv2.resize(cv2.imread(
						self.f_rgb[load_index]), (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)))
			raw_lidar.append(np.fromfile(
				self.f_lidar[load_index], dtype=np.float32).reshape((-1, 4)))
			if not self.is_testset:
				labels.append([line for line in open(
					self.f_label[load_index], 'r').readlines()])
			else:
				labels.append([''])
			tag.append(self.data_tag[load_index])
			voxel.append(process_pointcloud(raw_lidar[-1]))
			bsize, per_vox_feat, per_vox_num, per_vox_coor = build_input([voxel[-1]])
			vox_feature.append(per_vox_feat)
			vox_number.append(per_vox_num)
			vox_coordinate.append(per_vox_coor)

		#print('bsize', bsize)
		#rint('vox_feature', np.array(vox_feature).shape)
		#print('vox_number', vox_number)

		# Create augmented data
		augIdxs, aug_voxel = [], []
		if self.aug and not self.is_testset:
			num_to_load += self.aug_num
			# Get the indices of members to be augmented
			augIdxs = np.random.choice(np.arange(self.load_index, self.load_index+batch_size+1), 
												 self.aug_num, replace=False)
			for idx in augIdxs:
				ret = aug_data(self.data_tag[idx], self.object_dir)
				tag.append(ret[0])
				rgb.append(ret[1])
				raw_lidar.append(ret[2])
				aug_voxel.append(ret[3])
				labels.append(ret[4])

				bsize, per_vox_feat, per_vox_coor, per_vox_num = build_input([aug_voxel[-1]])
				vox_feature.append(per_vox_feat)
				#print('per_vox_feat', np.array(per_vox_feat).shape)
				vox_number.append(per_vox_num)
				vox_coordinate.append(per_vox_coor)

		#print('bsize', bsize)
		#print('vox_feature', np.array(vox_feature).shape)

		ret = (
				np.array(tag),
				np.array(labels),
				np.array(vox_feature),
				np.array(vox_number),
				np.array(vox_coordinate),
				np.array(rgb),
				np.array(raw_lidar)
			)

		flag = False
		self.load_index += batch_size
		if self.load_index >= self.dataset_size:
			flag = True
		return flag, ret
		





def build_input(voxel_dict_list):
	batch_size = len(voxel_dict_list)
	#print('batch_size', batch_size)

	feature_list = []
	number_list = []
	coordinate_list = []
	for i, voxel_dict in zip(range(batch_size), voxel_dict_list):
		feature_list.append(voxel_dict['feature_buffer'])
		number_list.append(voxel_dict['number_buffer'])
		coordinate = voxel_dict['coordinate_buffer']
		coordinate_list.append(
			np.pad(coordinate, ((0, 0), (1, 0)),
				   mode='constant', constant_values=i))

	#print('len(feature_list)', np.array(feature_list).shape)
	feature = np.concatenate(feature_list)
	#print('feature.shape', feature.shape)
	number = np.concatenate(number_list)
	coordinate = np.concatenate(coordinate_list)
	return batch_size, feature, number, coordinate
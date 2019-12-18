#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# helpers.py
# Contains helper functions just for the network
# G. Dylan Dickerson
# Created Dec 17, 2019
# Copied from https://github.com/tsinghua-rll/VoxelNet-tensorflow/

import tensorflow as tf
import numpy as np

def smooth_l1(deltas, targets, sigma=3.0):
	sigma2 = sigma * sigma
	diffs = tf.subtract(deltas, targets)
	smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

	smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
	smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
	smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + \
		tf.multiply(smooth_l1_option2, 1 - smooth_l1_signs)
	smooth_l1 = smooth_l1_add

	return smooth_l1


def build_input(voxel_dict_list):
	batch_size = len(voxel_dict_list)

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

	feature = np.concatenate(feature_list)
	number = np.concatenate(number_list)
	coordinate = np.concatenate(coordinate_list)
	return batch_size, feature, number, coordinate

def average_gradients(tower_grads):
	# ref:
	# https://github.com/tensorflow/models/blob/6db9f0282e2ab12795628de6200670892a8ad6ba/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L103
	# but only contains grads, no vars
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		grads = []
		for g in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			expanded_g = tf.expand_dims(g, 0)

			# Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)

		# Average over the 'tower' dimension.
		grad = tf.concat(axis=0, values=grads)
		grad = tf.reduce_mean(grad, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		grad_and_var = grad
		average_grads.append(grad_and_var)
	return average_grads
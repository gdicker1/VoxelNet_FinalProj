#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# layers.py
#  Define custom layers for VoxelNet
#  Created by G. Dylan Dickerson on 12 Dec 2019
#  Based on group_pointcloud.py and rpn.py at https://github.com/tsinghua-rll/VoxelNet-tensorflow

import numpy as np
import tensorflow as tf

from config import cfg

class VFELayer(object):
	def __init__(self, out_channels, name):
		super(VFELayer, self).__init__()
		self.units = int(out_channels / 2)
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
			self.dense = tf.layers.Dense(
				self.units, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
			self.batch_norm = tf.layers.BatchNormalization(
				name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)


	def apply(self, inputs, mask, training):
		# [K, T, 7] tensordot [7, units] = [K, T, units]
		pointwise = self.batch_norm.apply(self.dense.apply(inputs), training)

		#n [K, 1, units]
		aggregated = tf.reduce_max(pointwise, axis=1, keep_dims=True)

		# [K, T, units]
		repeated = tf.tile(aggregated, [1, cfg.POINT_PER_VOX, 1])

		# [K, T, 2 * units]
		concatenated = tf.concat([pointwise, repeated], axis=2)

		mask = tf.tile(mask, [1, 1, 2 * self.units])

		concatenated = tf.multiply(concatenated, tf.cast(mask, tf.float32))

		return concatenated


def ConvMD(M, Cin, Cout, k, s, p, input, training=True, activation=True, name='conv'):
	temp_p = np.array(p)
	temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
	with tf.variable_scope(name) as scope:
		if(M == 2):
			paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
			pad = tf.pad(input, paddings, "CONSTANT")
			temp_conv = tf.layers.conv2d(
				pad, Cout, k, strides=s, padding="valid", reuse=tf.AUTO_REUSE, name=scope)
		if(M == 3):
			paddings = (np.array(temp_p)).repeat(2).reshape(5, 2)
			pad = tf.pad(input, paddings, "CONSTANT")
			temp_conv = tf.layers.conv3d(
				pad, Cout, k, strides=s, padding="valid", reuse=tf.AUTO_REUSE, name=scope)
		temp_conv = tf.layers.batch_normalization(
			temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
		if activation:
			return tf.nn.relu(temp_conv)
		else:
			return temp_conv


def Deconv2D(Cin, Cout, k, s, p, input, training=True, name='deconv'):
	# Cin = channels in
	# Cout = channels out
	# k = kernel size
	# s = strides
	# p = padding
	temp_p = np.array(p)
	temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
	paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
	pad = tf.pad(input, paddings, "CONSTANT")
	with tf.variable_scope(name) as scope:
		temp_conv = tf.layers.conv2d_transpose(
			pad, Cout, k, strides=s, padding="SAME", reuse=tf.AUTO_REUSE, name=scope)
		temp_conv = tf.layers.batch_normalization(
			temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
		return tf.nn.relu(temp_conv)


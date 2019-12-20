#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# networks.py
# Implements the network parts for VoxelNet
# by G. Dylan Dickerson
# Created Dec. 17, 2019

import tensorflow as tf
import time

from config import cfg
from utils import *
from model.layers import VFELayer, ConvMD, Deconv2D
from model.helpers import smooth_l1, build_input

small_addon_for_BCE = 1e-6 # small add-on for Binary Cross Entropy

class FeatureNet(object):
	def __init__(self, training, batch_size, name=''):
		super(FeatureNet, self).__init__()
		self.training = training
		self.batch_size = batch_size

		self.feature = tf.placeholder(tf.float32, [None, cfg.POINT_PER_VOX, 7], name='feature')
		self.number = tf.placeholder(tf.int32, [None], name='number')
		self.coordinate = tf.placeholder(tf.int32, [None, 4], name='coordinate')

		with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
			self.vfe1 = VFELayer(32, 'VFE-1')
			self.vfe2 = VFELayer(128, 'VFE-2')
			self.dense = tf.layers.Dense(128, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
			self.batch_norm = tf.layers.BatchNormalization(name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)
		
		mask = tf.not_equal(tf.reduce_max(self.feature, axis=2, keep_dims=True), 0)
		mask = tf.not_equal(tf.reduce_max(self.feature, axis=2, keep_dims=True), 0)
		temp = self.vfe1.apply(self.feature, mask, self.training)
		temp = self.vfe2.apply(temp, mask, self.training)
		temp = self.dense.apply(temp)
		temp = self.batch_norm.apply(temp, self.training)

		out = tf.reduce_max(temp, axis=1)

		self.outputs = tf.scatter_nd(self.coordinate, out, [self.batch_size, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])


class ConvMiddleNet(object):
	def __init__(self, input, training=True, name=''):
		self.input = input
		self.training = training

		with tf.variable_scope('ConvMiddle_' + name):
			temp_conv = ConvMD(3, 128, 64, 3, (2, 1, 1), (1,1,1),
							   self.input, name='mid_conv1')
			temp_conv = ConvMD(3, 64, 64, 3, (1, 1, 1), (0,1,1),
							   temp_conv, name='mid_conv2')
			temp_conv = ConvMD(3, 64, 64, 3, (2, 1, 1), (1,1,1),
							   temp_conv, name='mid_conv3')
			temp_conv = tf.transpose(temp_conv, perm=[0, 2, 3, 4, 1])
			temp_conv = tf.reshape(temp_conv, [-1, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])
			self.outputs = temp_conv
			self.output_shape = [cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128]
		return

class RPN(object):
	def __init__(self, input, alpha=1.5, beta=1, sigma=3, training=True, name=''):
		self.input = input
		self.training = training
		# Targets are ground-truth boxes, (delx, dely, delz, dell , delw, delh, rotation)
		self.targets = tf.placeholder(tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 14])
		# Positive anchors (matching anchors)
		self.pos_equal_one = tf.placeholder(tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2])
		self.pos_equal_one_sum = tf.placeholder(tf.float32, [None, 1, 1, 1])
		self.pos_equal_one_for_reg = tf.placeholder(tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 14])
		# Negative anchors (non-matching anchors)
		self.neg_equal_one = tf.placeholder(tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2])
		self.neg_equal_one_sum = tf.placeholder(tf.float32, [None, 1, 1, 1])

		with tf.variable_scope('RPN_' + name):
			# Block 1
			#  2D Conv, 128 channels in, 128 channels out, 3x3 kernel, stride 2, padding 1
			#  stride two halves the size of the input
			temp = ConvMD(2, 128, 128, 3, (2,2), (1,1), self.input, training=self.training, name='rpn_conv1')
			#  2D Conv, 128 channels in, 128 channels out, 3x3 kernel, stride 1, padding 1
			temp = ConvMD(2, 128, 128, 3, (1,1), (1,1), temp, training=self.training, name='rpn_conv2')
			temp = ConvMD(2, 128, 128, 3, (1,1), (1,1), temp, training=self.training, name='rpn_conv3')
			temp = ConvMD(2, 128, 128, 3, (1,1), (1,1), temp, training=self.training, name='rpn_conv4')
			#  "Upsample". 128 channels in, 256 out, 3x3 kernel, stride 1, padding 0
			#    output of this layer is [orig_input_height/2, orig_input_width/2, 256]
			deconv1 = Deconv2D(128, 256, 3, (1,1), (0,0), temp, training=self.training, name='rpn_deconv1')
			
			# Block 2
			#  2D Conv, 128 channels in, 128 channels out, 3x3 kernel, stride 2, padding 1
			#  stride two halves the size of the input
			temp = ConvMD(2, 128, 128, 3, (2,2), (1,1), temp, training=self.training, name='rpn_conv5')
			#  2D Conv, 128 channels in, 128 channels out, 3x3 kernel, stride 1, padding 1
			temp = ConvMD(2, 128, 128, 3, (1,1), (1,1), temp, training=self.training, name='rpn_conv6')
			temp = ConvMD(2, 128, 128, 3, (1,1), (1,1), temp, training=self.training, name='rpn_conv7')
			temp = ConvMD(2, 128, 128, 3, (1,1), (1,1), temp, training=self.training, name='rpn_conv8')
			temp = ConvMD(2, 128, 128, 3, (1,1), (1,1), temp, training=self.training, name='rpn_conv9')
			temp = ConvMD(2, 128, 128, 3, (1,1), (1,1), temp, training=self.training, name='rpn_conv10')
			#  "Upsample". 128 channels in, 256 out, 3x3 kernel, stride 2, padding 0
			#    output of this layer is [orig_input_height/2, orig_input_width/2, 256]
			deconv2 = Deconv2D(128, 256, 2, (2,2), (0,0), temp, training=self.training, name='rpn_deconv2')

			# Block 3
			#  2D Conv, 128 channels in, 128 channels out, 3x3 kernel, stride 2, padding 1
			#  stride two halves the size of the input
			temp = ConvMD(2, 128, 256, 3, (2,2), (1,1), temp, training=self.training, name='rpn_conv11')
			#  2D Conv, 128 channels in, 128 channels out, 3x3 kernel, stride 1, padding 1
			temp = ConvMD(2, 256, 256, 3, (1,1), (1,1), temp, training=self.training, name='rpn_conv12')
			temp = ConvMD(2, 256, 256, 3, (1,1), (1,1), temp, training=self.training, name='rpn_conv13')
			temp = ConvMD(2, 256, 256, 3, (1,1), (1,1), temp, training=self.training, name='rpn_conv14')
			temp = ConvMD(2, 256, 256, 3, (1,1), (1,1), temp, training=self.training, name='rpn_conv15')
			temp = ConvMD(2, 256, 256, 3, (1,1), (1,1), temp, training=self.training, name='rpn_conv16')
			#  "Upsample". 128 channels in, 256 out, 3x3 kernel, stride 4, padding 0
			#    output of this layer is [orig_input_height/2, orig_input_width/2, 256]
			deconv3 = Deconv2D(256, 256, 4, (4,4), (0,0), temp, training=self.training, name='rpn_deconv3')

			# Concat
			#  Creates a block [orig_input_height/2, orig_input_width/2, 256*3=768]
			temp = tf.concat([deconv3, deconv2, deconv1], -1)

			# Outputs
			#  Probability score map that gives likelihood of class
			#  Perform 1x1 depth wise convolution and only have [orig_input_height/2, orig_input_width/2, 2] as output
			cls_map = ConvMD(2, 768, 2, 1, (1,1), (0,0), temp, activation=False, training=self.training, name='rpn_conv17')
			self.p_pos = tf.sigmoid(cls_map)
			#  Regression map that gives bounding box info
			#  Performs 1x1 depth wise convolution and only have [orig_input_height/2, orig_input_width/2, 14] as output
			reg_map = ConvMD(2, 768, 14, 1, (1,1), (0,0), temp, activation=False, training=self.training, name='rpn_conv18')
			self.output_shape = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]

			# Loss
			self.cls_loss = alpha * (-self.pos_equal_one * tf.log(self.p_pos + small_addon_for_BCE)) / self.pos_equal_one_sum \
				+ beta * (-self.neg_equal_one * tf.log(1 - self.p_pos +
													   small_addon_for_BCE)) / self.neg_equal_one_sum
			self.cls_loss = tf.reduce_sum(self.cls_loss)

			self.reg_loss = smooth_l1(reg_map * self.pos_equal_one_for_reg, self.targets * self.pos_equal_one_for_reg, sigma) / self.pos_equal_one_sum
			self.reg_loss = tf.reduce_sum(self.reg_loss)

			self.loss = tf.reduce_sum(self.cls_loss + self.reg_loss)

			self.delta_output = reg_map
			self.prob_output = self.p_pos
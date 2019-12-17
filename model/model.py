#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# model.py
# Contains helper functions just for the network
# G. Dylan Dickerson
# Created Dec 17, 2019

import tensorflow as tf
import time

from config import cfg
from utils import *
from model.networks import FeatureNet, ConvMiddleNet, RPN

class VoxelNet(object):
	def __init__(self, cls='Car', single_batch_size=2, learning_rate=0.001,
				 max_gradient_norm = 5.0, alpha=1.5, beta=1, is_train=True, avail_gpus=['0']):
		# Initialze self variables
		self.cls = cls
		self.single_batch_size = single_batch_size
		self.learning_rate = tf.Variable(float(learning_rate, trainable=False, dtype=tf.float32))
		self.global_step = tf.Variable(1, trainable=False)
		self.epoch = tf.Variable(0, trainable=False)
		self.epoch_add_op = self.epoch.assign(self.epoch + 1)
		self.alpha = alpha
		self.beta = beta
		self.avail_gpus = avail_gpus

		# Set learning rate syllabus
		lr = tf.train.exponential_decay(self.learning_rate, self.global_step, 10000, 0.96)
		self.opt = tf.train.AdamOptimizer(lr)

		# build graph
		self.vox_feature = []
		self.vox_number = []
		self.vox_coordinate = []
        self.targets = []
        self.pos_equal_one = []
        self.pos_equal_one_sum = []
        self.pos_equal_one_for_reg = []
        self.neg_equal_one = []
        self.neg_equal_one_sum = []
        self.delta_output = []
        self.prob_output = []
        self.gradient_norm = []
        self.tower_grads = []

        with tf.variable_scope(tf.get_variable_scope()):
        	for idx, dev in enumerate(self.avail_gpus):
        		with tf.device('/gpu:{}'.format(dev)), tf.name_scope('gpu{}'.format(dev)):
        			feature = FeatureNet(training=is_train, batch_size=single_batch_size)
        			mid = ConvMiddleNet(feature.outputs, training=is_train)
        			rpn = RPN(mid.outputs, alpha, beta, sigma, training=is_train)
        			tf.get_variable_scope().reuse_variables()

        			# Input
        			self.vox_feature.append(feature.feature)
        			self.vox_number.append(feature.number)
        			self.vox_coordinate.append(feature.coordinate)
        			self.targets.append(rpn.targets)
        			self.pos_equal_one.append(rpn.pos_equal_one)
        			self.pos_equal_one_sum.append(rpn.pos_equal_one_sum)
        			self.pos_equal_one_for_reg.append(rpn.pos_equal_one_for_reg)
        			self.neg_equal_one.append(rpn.neg_equal_one)
        			self.neg_equal_one_sum.append(rpn.neg_equal_one_sum)

        			# Loss and train
        			self.loss = rpn.loss
        			self.reg_loss = rpn.reg_loss
        			self.cls_loss = rpn.cls_loss
        			self.params = tf.trainable_variables()
        			gradients = tf.gradients(self.loss, self.params)
        			clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)

        			# Output
        			feature_output = feature.outputs
        			self.delta_output.append(rpn.delta_output)
        			self.prob_output.append(rpn.prob_output)
        			self.tower_grads.append(clipped_gradients)
        			self.gradient_norm.append(gradient_norm)
        			self.rpn_output_shape = rpn.output_shape
		with tf.device('/gpu:{}'.format(self.avail_gpus[0])):
			self.grads = average_gradients(self.tower_grads)
			self.update = self.opt.apply_gradients(zip(self.grads, self.params), global_step=self.global_step)
			self.gradient_norm = tf.group(*self.gradient_norm)

		self.delta_output = tf.concat(self.delta_output, axis=0)
		self.prob_output = tf.concat(self.prob_output, axis = 0)

		self.anchors = cal_anchors()

		self.rbg = tf.placeholder(tf.uint8, [None, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3])
		self.bv_heatmap = tf.placeholder(tf.uint8, [None, cfg.BV_LOG_FACTOR * cfg.FEATURE_HEIGHT, cfg.BV_LOG_FACTOR * cfg.FEATURE_WIDTH, 3])
        self.boxes2d = tf.placeholder(tf.float32, [None, 4])
        self.boxes2d_scores = tf.placeholder(tf.float32, [None])

        # NMS(2D)
        with tf.device('/gpu:{}'.format(self.avail_gpus[0])):
            self.box2d_ind_after_nms = tf.image.non_max_suppression(
                self.boxes2d, self.boxes2d_scores, max_output_size=cfg.RPN_NMS_POST_TOPK, iou_threshold=cfg.RPN_NMS_THRESH)

        # summary and saver
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                    max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('train/loss', self.loss),
            tf.summary.scalar('train/reg_loss', self.reg_loss),
            tf.summary.scalar('train/cls_loss', self.cls_loss),
            *[tf.summary.histogram(each.name, each) for each in self.params]
        ])

        self.validate_summary = tf.summary.merge([
            tf.summary.scalar('validate/loss', self.loss),
            tf.summary.scalar('validate/reg_loss', self.reg_loss),
            tf.summary.scalar('validate/cls_loss', self.cls_loss)
        ])

        self.predict_summary = tf.summary.merge([
            tf.summary.image('predict/bird_view_lidar', self.bv),
            tf.summary.image('predict/bird_view_heatmap', self.bv_heatmap),
            tf.summary.image('predict/front_view_rgb', self.rgb),
        ])


    def train_step(self, session, data, train=False, summary=False):
    	tag = data[0]
    	label = data[1]
    	vox_feature = data[2]
    	vox_number = data[3]
    	vox_coordinate = data[4]
    	#print('train', tag)
    	pos_equal_one, neg_equal_one, targets = cal_rpn_target(label, self.rpn_output_shape,
    														   self.anchors, cls=cfg.TARGET_OBJ, coordinate='lidar')
    	pos_equal_one_for_reg = np.concatenate([np.tile(pos_equal_one[...,[0]], 7), np.tile(pos_equal_one[...,[1]], 7)], axis=-1)
    	pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
    	neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)

    	input_feed = {}
    	for idx in range(len(self.avail_gpus)):
    		input_feed[self.vox_feature[idx]] = vox_feature[idx]
            input_feed[self.vox_number[idx]] = vox_number[idx]
            input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]
            input_feed[self.targets[idx]] = targets[idx *
                                                    self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one[idx]] = pos_equal_one[idx *
                                                                self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_sum[idx]] = pos_equal_one_sum[idx *
                                                                        self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_for_reg[idx]] = pos_equal_one_for_reg[idx *
                                                                                self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one[idx]] = neg_equal_one[idx *
                                                                self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one_sum[idx]] = neg_equal_one_sum[idx *
                                                                        self.single_batch_size:(idx + 1) * self.single_batch_size]

        if train:
        	output_feed = [self.loss, self.reg_loss, self.cls_loss, self.gradient_norm, self.update]
    	else:
    		output_feed = [self.loss, self.reg_loss, self.cls_loss]
		if summary:
			output_feed.append(self.train_summary)

		return session.run(output_feed, input_feed)


	def validate_step(self, session, data, summary=False):
		tag = data[0]
    	label = data[1]
    	vox_feature = data[2]
    	vox_number = data[3]
    	vox_coordinate = data[4]
    	#print('valid', tag)

    	pos_equal_one, neg_equal_one, targets = cal_rpn_target(label, self.rpn_output_shape,
    														   self.anchors, cls=cfg.TARGET_OBJ, coordinate='lidar')
    	pos_equal_one_for_reg = np.concatenate([np.tile(pos_equal_one[...,[0]], 7), np.tile(pos_equal_one[...,[1]], 7)], axis=-1)
    	pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
    	neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)

    	input_feed = {}
    	for idx in range(len(self.avail_gpus)):
    		input_feed[self.vox_feature[idx]] = vox_feature[idx]
            input_feed[self.vox_number[idx]] = vox_number[idx]
            input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]
            input_feed[self.targets[idx]] = targets[idx *
                                                    self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one[idx]] = pos_equal_one[idx *
                                                                self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_sum[idx]] = pos_equal_one_sum[idx *
                                                                        self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_for_reg[idx]] = pos_equal_one_for_reg[idx *
                                                                                self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one[idx]] = neg_equal_one[idx *
                                                                self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one_sum[idx]] = neg_equal_one_sum[idx *
                                                                        self.single_batch_size:(idx + 1) * self.single_batch_size]

        output_feed = [self.loss, self.reg_loss, self.cls_loss]
        if summary:
			output_feed.append(self.train_summary)

		return session.run(output_feed, input_feed)


	def predict_step(self, session, data, summary=False):
		tag = data[0]
    	label = data[1]
    	vox_feature = data[2]
    	vox_number = data[3]
    	vox_coordinate = data[4]
    	img = data[5]
    	lidar=data[6]

    	if summary:
    		batch_gt_boxes3d = label_to_gt_box3d(label, cls=self.cls, coordinate='lidar')
		#print('predict', tag)

		input_feed = {}
		for idx in range(len(self.avail_gpus)):
			input_feed[self.vox_feature[idx]] = vox_feature[idx]
			input_feed[self.vox_number[idx]] = vox_number[idx]
			input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]

		output_feed = [self.prob_output, self.delta_output]
		probs, deltas = session.run(output_feed, input_feed)

		batch_boxes3d = delta_to_boxes3d(deltas, self.anchors, coordinate='lidar')
		batch_boxes2d = batch_boxes3d[:,:,[0,1,4,5,6]]
		batch_probs = probs.reshape(len(self.avail_gpus) * self.single_batch_size, -1)

		# Non max suppression
		ret_box3d = []
		ret_score = []
		for batch_id in range(len(self.avail_gpus) * self.single_batch_size):
			ind = np.where(batch_probs[batch_id, :] >= cfg.RPN_SCORE_THRESH)[0]
			tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
			tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
			tmp_scores = batch_probs[batch_id, ind]

			boxes2d = corner_to_standup_box2d(center_to_corner_box2d(tmp, boxes2d, coordinate='lidar'))
			ind = session.run(self.box2d_ind_after_nms, {self.boxes2d: boxes2d, self.boxes2d_scores: tmp_scores})
			tmp_boxes3d = tmp_boxes3d[ind, ...]
			tmp_scores = tmp_scores[ind]
			ret_box3d.append(tmp_boxes3d)
			ret_score.append(tmp_scores)

		ret_box3d_score = []
		for boxes3d, scores in zip(ret_box3d, ret_score):
			ret_box3d_score.append(np.concatenate([np.tile(self.cls, len(boxes3d))[:, np.newaxis],
                                                   boxes3d, scores[:, np.newaxis]], axis=-1))

		if summary:
			front_image = draw_lidar_box3d_on_image(img[0], ret_box3d[0], ret_score[0], batch_gt_boxes3d[0])
			bird_view = lidar_to_bird_view_img(lidar[0], factor=cfg.BV_LOG_FACTOR)
			bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[0], ret_score[0], batch_gt_boxes3d[0], factor=cfg.BV_LOG_FACTOR)
			heatmap = colorize(probs[0,...], cfg.BV_LOG_FACTOR)
			ret_summary = session.run(self.predict_summary, 
				                      {self.rgb: front_image[np.newaxis, ...],
				                       self.bv: bird_view[np.newaxis, ...],
				                       self.bv_heatmap: heatmap[np.newaxis, ...]})

			return tag, ret_box3d_score, ret_summary
		return tag, ret_box3d_score

    def forwardPass(self, session, data):
        """Perform a single forward pass of the network and get timing"""
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        img = data[5]
        lidar=data[6]

        input_feed = {}
        for idx in range(len(self.avail_gpus)):
            input_feed[self.vox_feature[idx]] = vox_feature[idx]
            input_feed[self.vox_number[idx]] = vox_number[idx]
            input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]

        output_feed = [self.prob_output, self.delta_output]
        preRunTime = time.time()
        probs, deltas = session.run(output_feed, input_feed)
        postRunTime = time.time()
        return postRunTime - preRunTime
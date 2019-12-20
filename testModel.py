#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# testModel.py
# loads the parameters from a file, checks time for forward pass, and average precision
# by G. Dylan Dickerson
# Created 18 Dec 2019

import argparse
import os
import time
import sys
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from config import cfg
from model import VoxelNet
from utils import *
from utils.kitti_loader_sequential import KittiLoaderSeq as KittiLoader

def getTotalNumberOfParams(model):
	"""Find out and print the total number of variables in the model"""
	totalParams = 0
	for i, variable in enumerate(model.params):
		shape = variable.get_shape()
		print("Shape {}:".format(i), shape)
		print("Shape {} len:".format(i), len(shape))
		variable_params = 1
		for j,dim in enumerate(shape):
			print("dim {} in shape {}".format(j,i), dim)
			variable_params *= dim.value
		print("Shape {} has a total of {} trainable parameters".format(i, variable_params))
		totalParams += variable_params
	print("Model has a total of {} trainable parameters".format(totalParams))

def getConfusionMatrix(cls_scores, maxIoUs, score_thresh, cls='Car', noGroundTruth=False):
	true_neg, true_pos, false_neg, false_pos = 0, 0, 0, 0

	if noGroundTruth and len(cls_scores) > 0:
		for i in range(len(cls_scores)):
			if cls_scores[i] < score_thresh:
				false_pos += 1
	elif not noGroundTruth:
		for i in range(len(cls_scores)):
			if cls_scores[i] >= score_thresh and maxIoUs[i] >= 0.7:
				true_pos += 1
			elif cls_scores[i] >= score_thresh and maxIoUs[i] < 0.7:
				false_pos += 1
			elif cls_scores[i] < score_thresh and maxIoUs[i] >= 0.7:
				true_neg += 1
			elif cls_scores[i] < score_thresh and maxIoUs[i] < 0.7:
				false_neg += 1

	return np.array([true_neg, true_pos, false_neg, false_pos])

# TODO make sure this implemented correctly if other functions ever are
# def getPrecisionAndRecall(session, model, data, threshold):
# 	times, true_negative, true_positive, false_positive, false_negative = [], [], [], [], []
# 	ious = []
# 	confMat = np.zeros(4)
# 	for i, dat in enumerate(data):
# 		if i >= 1:
# 			break
# 		labels = dat[1]
# 		# Run the model to get time for forward pass, probability map, and regression map
# 		t_time, prob, reg = model.forwardPass(session, dat)
# 		times.append(t_time)

# 		# Get the prediction into bounding box form
# 		batch_boxes3d = delta_to_boxes3d(reg, model.anchors, coordinate='lidar')
# 		batch_boxes2d = batch_boxes3d[:,:,[0,1,4,5,6]]
# 		batch_boxes2d_standup = [corner_to_standup_box2d(center_to_corner_box2d(batch_boxes2d[i], coordinate='lidar')) for i in len(batch_boxes2d)]
# 		batch_probs = probs.reshape(len(model.avail_gpus) * model.single_batch_size, -1)

# 		# Non max suppression (limit the number of boxes)
# 		ret_box3d = []
# 		ret_score = []
# 		for batch_id in range(1):
# 			ind = np.where(batch_probs[batch_id, :] >= cfg.RPN_SCORE_THRESH)[0]
# 			tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
# 			tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
# 			tmp_scores = batch_probs[batch_id, ind]

# 			boxes2d = corner_to_standup_box2d(center_to_corner_box2d(tmp_boxes2d, coordinate='lidar'))
# 			ind = session.run(model.box2d_ind_after_nms, {model.boxes2d: boxes2d, model.boxes2d_scores: tmp_scores})
# 			tmp_boxes3d = tmp_boxes3d[ind, ...]
# 			tmp_scores = tmp_scores[ind]
# 			ret_box3d.append(tmp_boxes3d)
# 			ret_score.append(tmp_scores)

# 		batch_size = labels.shape[0]
# 		batch_gt_boxes3d = label_to_gt_box3d(labels, cls='Car', coordinate=coordinate)

# 		ious = cal_box3d_iou(ret_box3d[0], batch_gt_boxes3d)
# 		maxIous = np.amax(ious, axis=1)

# 		confMat += getConfusionMatrix(ret_score[0], maxIoUs, threshold)

# 	precision = confMat[3] / (confMat[1] + confMat[3])
# 	recall = confMat[2] / (confMat[2] + confMat[3])

# 	return times, precision, recall

def getPrecisionAndRecall(session, model, loader, threshold):
	times, true_negative, true_positive, false_positive, false_negative = [], [], [], [], []
	ious = []
	confMat = np.zeros(4)
	for i in range(len(loader)):
		flag, dat = loader.load(1)
		labels = dat[1]

		# Run the model to get time for forward pass, probability map, and regression map
		t_time, prob, reg = model.forwardPass(session, dat)
		times.append(t_time)
		print('iteration {} took {} secs for forwardPass'.format(i, t_time))
		if i >= 1005:
			break
		else:
			continue

		# Get the prediction into bounding box form
		batch_boxes3d = delta_to_boxes3d(reg, model.anchors, coordinate='lidar')
		batch_boxes2d = batch_boxes3d[:,:,[0,1,4,5,6]]
		batch_boxes2d_standup = [corner_to_standup_box2d(center_to_corner_box2d(batch_boxes2d[i], coordinate='lidar')) for i in range(len(batch_boxes2d))]
		batch_probs = prob.reshape(len(model.avail_gpus) * model.single_batch_size, -1)

		# Non max suppression (limit the number of boxes)
		ret_box3d = []
		ret_score = []
		for batch_id in range(1):
			ind = np.where(batch_probs[batch_id, :] >= cfg.RPN_SCORE_THRESH)[0]
			tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
			tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
			tmp_scores = batch_probs[batch_id, ind]

			boxes2d = corner_to_standup_box2d(center_to_corner_box2d(tmp_boxes2d, coordinate='lidar'))
			ind = session.run(model.box2d_ind_after_nms, {model.boxes2d: boxes2d, model.boxes2d_scores: tmp_scores})
			tmp_boxes3d = tmp_boxes3d[ind, ...]
			tmp_scores = tmp_scores[ind]
			ret_box3d.append(tmp_boxes3d)
			ret_score.append(tmp_scores)

		batch_size = labels.shape[0]
		batch_gt_boxes3d = np.array(label_to_gt_box3d(labels, cls='Car', coordinate='lidar'))
		if batch_gt_boxes3d.size == 0:
			print('no labels in {} class in labels'.format('Car'))
			confMat += getConfusionMatrix(ret_score[0], [], threshold, noGroundTruth=True)			
			continue
		ret_box3d = np.array(ret_box3d)
		
		ious = cal_box3d_iou(ret_box3d[0], batch_gt_boxes3d[0])
		maxIoUs = np.amax(ious, axis=1)
		#print(maxIoUs)

		confMat += getConfusionMatrix(ret_score[0], maxIoUs, threshold, noGroundTruth=False)

	loader.reset()
	return times, 0.0, 0.0
	precision = confMat[3] / (confMat[1] + confMat[3])
	recall = confMat[2] / (confMat[2] + confMat[3])

	return times, precision, recall

def plotPrecisionRecallCurve(precisions, recalls, filename='precision_recall_curve.png'):
	fig = plt.figure()
	ax = fig.axes
	
	plt.plot(precisions[0], recalls[0], 'go-')

	plt.title('Precision-Recall Curve with {} bins'.format(cfg.THRESHOLD_BINS))

	plt.savefig(filename)

def plotPrecisionRecallCurves(precisions, recalls, filename='precision_recall_curve.png'):
	fig = plt.figure()
	ax = fig.axes
	
	plt.plot(precisions[0], recalls[0], 'go-', label='easy')
	plt.plot(precisions[1], recalls[1], 'yD-', label='moderate')
	plt.plot(precisions[2], recalls[2], 'r^-', label='hard')

	plt.legend(frameon=False)

	plt.title('Precision-Recall Curves with {} bins'.format(cfg.THRESHOLD_BINS))

	plt.savefig(filename)

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.ERROR)
	parser = argparse.ArgumentParser(description='test network')
	parser.add_argument('-c', '--cls', type=str, default='Car',
						help='type of object is being classified (Car, Pedestrian, Cyclist)')
	parser.add_argument('-d', '--save-model-dir', type=str, default=os.path.join(os.getcwd(), 'checkpoint', 'default'),
						help='directory to load the model from')
	args = parser.parse_args()

	save_model_dir = args.save_model_dir
	dataset_dir = cfg.DATA_DIR

	with tf.Graph().as_default():
		with KittiLoader(object_dir=os.path.join(dataset_dir, 'object', 'training'), require_shuffle=False,
						 split_file=os.path.join(cfg.ROOT_DIR, 'DataSplits', 'val.txt'),
			             is_testset=False, batch_size=1) as valid_loader:
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
												visible_device_list=cfg.GPU_AVAILABLE,
												allow_growth=True)

			config = tf.ConfigProto(gpu_options=gpu_options, device_count={"GPU": cfg.GPU_USE_COUNT}, allow_soft_placement=True)
			with tf.Session(config=config) as sess:
				premodelTime = time.time()
				model = VoxelNet(cls=args.cls, single_batch_size=1,
								 is_train=False, avail_gpus=cfg.GPU_AVAILABLE.split(','))
				postmodelTime = time.time()
				#getTotalNumberOfParams(model)
				print("It took {} seconds to create model".format(postmodelTime - premodelTime))
				print("Reading model parameters from ", save_model_dir)
				prereadTime = time.time()
				model.saver.restore(sess, tf.train.latest_checkpoint(save_model_dir))
				postreadTime = time.time()
				print("It took {} seconds to read parameters from file".format(postreadTime - prereadTime))

				# TODO properly split data into easy, moderate, and hard
				#preLoad = time.time()
				#print('Loading data for each mode')
				#easy_data = valid_loader.load_mode('easy')
				#print('Loaded easy')
				#mod_data = valid_loader.load_mode('moderate')
				#print('Loaded moderate')
				#hard_data = valid_loader.load_mode('hard')
				#print('Loaded hard')
				#postLoad = time.time()
				#print("It took {} seconds to load data for all modes".format(postLoad-preLoad))

				# print('There are {} data points in easy mode'.format(len(easy_data)))
				# print('There are {} data points in moderate mode'.format(len(mod_data)))
				# print('There are {} data points in hard mode'.format(len(hard_data)))

				threshold_values = np.linspace(0.0, 1.0, num=cfg.THRESHOLD_BINS)
				#precisions = np.zeros((3,cfg.THRESHOLD_BINS))
				#ecalls = np.zeros((3,cfg.THRESHOLD_BINS))
				precisions = np.zeros(cfg.THRESHOLD_BINS)
				recalls = np.zeros(cfg.THRESHOLD_BINS)
				times = []

				for i,tval in enumerate(range(cfg.THRESHOLD_BINS)):
					print('Getting precisions and recalls for thresh={} and iter={}'.format(tval, i))
					tim, precisions[i], recalls[i] = getPrecisionAndRecall(sess, model, valid_loader, tval)
					times.append(tim)
					print(times)
					break

				# # Find precision and recall for easy_data
				# for i,tval in range(cfg.THRESHOLD_BINS):
				# 	tim, precisions[0,i], recalls[0,i] = getPrecisionAndRecall(sess, model, easy_data, threshold_values[i])
				# 	times.append(tim)

				# # Find precision and recall for moderate_data
				# for i,tval in range(cfg.THRESHOLD_BINS):
				# 	tim, precisions[1,i], recalls[1,i] = getPrecisionAndRecall(sess, model, easy_data, threshold_values[i])
				# 	times.append(tim)

				# # Find precision and recall for hard_data
				# for i,tval in range(cfg.THRESHOLD_BINS):
				# 	tim, precisions[2,i], recalls[2,i] = getPrecisionAndRecall(sess, model, easy_data, threshold_values[i])
				# 	times.append(tim)

				times = np.array(times).flatten()
				orig_forward_passes = len(times)
				times = times[5:] # skip first five because of warmup
				num_forward_passes = len(times)
				ave_time = np.average(times)
				std_time = np.std(times)

				print('precisions', precisions)

				plotPrecisionRecallCurves(precisions, recalls)
				average_precisions = np.average(precisions)

				#print('For easy data achieved average precision of {}'.format(average_precisions[0]))
				#print('For moderate data achieved average precision of {}'.format(average_precisions[1]))
				#print('For hard data achieved average precision of {}'.format(average_precisions[2]))
				print('Average precision is {}'.format(average_precisions))
				print('Forward passes took an average of {} +/- {} seconds over {} of {} forward passes'.format(ave_time, std_time, num_forward_passes, orig_forward_passes))
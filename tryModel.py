#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# testModel.py
# Do some forward passes of the model to make sure that it works
# By G. Dylan Dickerson
# Created 17 Dec 2019

import argparse
import os
import time
import sys
import tensorflow as tf
import time
from numpy.random import randint

from config import cfg
from model import VoxelNet
from utils.kitti_loader_sequential import KittiLoaderSeq as KittiLoader

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--numPasses', type=int, default=1,
					help='number of forward passes to do')
parser.add_argument('-b', '--single-batch-size', type=int, default=1,
					help='set batch size for each GPU')
parser.add_argument('-c', '--cls', type=str, default='Car',
					help='type of object is being classified (Car, Pedestrian, Cyclist)')
args = parser.parse_args()

dataset_dir = cfg.DATA_DIR
label_dir = os.path.join(dataset_dir, 'object', 'training', 'label_2')


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


def main(_):
	with tf.Graph().as_default():
		with KittiLoader(object_dir=os.path.join(dataset_dir, 'object', 'training'), require_shuffle=True,
						 split_file=os.path.join(cfg.ROOT_DIR, 'DataSplits', 'train.txt'),
						 is_testset=False, batch_size=args.single_batch_size, aug=False, aug_num=2) as valid_loader:
			valid_loader.reset()
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
					visible_device_list=cfg.GPU_AVAILABLE,
					allow_growth=True)
			config = tf.ConfigProto(gpu_options=gpu_options, device_count={"GPU": cfg.GPU_USE_COUNT}, allow_soft_placement=True)

			with tf.Session(config=config) as sess:
				premodelTime = time.time()
				model = VoxelNet(cls=args.cls, single_batch_size=args.single_batch_size, max_gradient_norm=5.0,
												 is_train=True, alpha=1.5, beta=1, avail_gpus=cfg.GPU_AVAILABLE.split(','))
				postmodelTime = time.time()
				getTotalNumberOfParams(model)
				print("It took {} seconds to create model".format(postmodelTime - premodelTime))

				print("Initializing model parameters")
				preInitTime = time.time()
				tf.global_variables_initializer().run()
				postInitTime = time.time()
				print("It took {} seconds to freshly initialize model parameters".format(postInitTime - preInitTime))

				numIter = 0
				numFiles = len([name for name in os.listdir(label_dir) if os.path.isfile(name)])
				while numIter < args.numPasses:
					print('Attempting to load {} members from dataset'.format(args.single_batch_size))
					flag, data = valid_loader.load(args.single_batch_size)
					if flag:
						valid_loader.reset()
					theTime, _, _ = model.forwardPass(sess, data)
					print('Forward pass took {} seconds'.format(theTime))
					numIter += 1

				return


if __name__ == '__main__':
	tf.app.run(main)

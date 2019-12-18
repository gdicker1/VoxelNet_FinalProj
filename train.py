#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# train.py
# Define the training process for VoxelNet
# by G. Dylan Dickerson
# Created 17 Dec 2019
# based on train.py from https://github.com/tsinghua-rll/VoxelNet-tensorflow

import argparse
import os
import time
import sys
import tensorflow as tf
import time

from config import cfg
from model import VoxelNet
#from utils.kitti_loader import KittiLoader
from utils.kitti_loader_sequential import KittiLoaderSeq as KittiLoader
from train_hook import check_if_should_pause

# Create parser for command line arguments
parser = argparse.ArgumentParser(description='training')
parser.add_argument('-i', '--max_epoch', type=int, default=10,
					help='maximum number of epochs to run for')
parser.add_argument('-n', '--tag', type=str, default='default',
					help='set log tag')
parser.add_argument('-b', '--single-batch-size', type=int, default=1,
					help='set batch size for each GPU')
parser.add_argument('-l', '--learning_rate', type=float, default=0.001,
					help='initial learning-rate for training')
parser.add_argument('-c', '--cls', type=str, default='Car',
					help='type of object is being classified (Car, Pedestrian, Cyclist)')
args = parser.parse_args()

# Setup / Create directories used
dataset_dir = cfg.DATA_DIR
log_dir = os.path.join(cfg.LOG_DIR, args.tag)
save_model_dir = os.path.join(cfg.CHECKPOINT_DIR, args.tag)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_model_dir, exist_ok=True)

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
		global save_model_dir
		with KittiLoader(object_dir=os.path.join(dataset_dir, 'object', 'training'), require_shuffle=True,
						 split_file=os.path.join(cfg.ROOT_DIR, 'DataSplits', 'train.txt'),
			             is_testset=False, batch_size=args.single_batch_size, aug=True, aug_num=1) as train_loader, \
			KittiLoader(object_dir=os.path.join(dataset_dir, 'object', 'training'), require_shuffle=False,
						 split_file=os.path.join(cfg.ROOT_DIR, 'DataSplits', 'val.txt'),
			             is_testset=False, batch_size=args.single_batch_size, aug=False, aug_num=0) as valid_loader:

			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
												visible_device_list=cfg.GPU_AVAILABLE,
												allow_growth=True)
			config = tf.ConfigProto(gpu_options=gpu_options, device_count={"GPU": cfg.GPU_USE_COUNT}, allow_soft_placement=True)
			with tf.Session(config=config) as sess:
				premodelTime = time.time()
				model = VoxelNet(cls=args.cls, single_batch_size=args.single_batch_size,
								 learning_rate=args.learning_rate, max_gradient_norm=5.0,
								 is_train=True, alpha=1.5, beta=1, avail_gpus=cfg.GPU_AVAILABLE.split(','))
				postmodelTime = time.time()
				getTotalNumberOfParams(model)
				print("It took {} seconds to create model".format(postmodelTime - premodelTime))

				# Restore from checkpoint if it exists
				if tf.train.get_checkpoint_state(save_model_dir):
					print("Reading model parameters from ", save_model_dir)
					prereadTime = time.time()
					model.saver.restore(sess, tf.train.latest_checkpoint(save_model_dir))
					postreadTime = time.time()
					print("It took {} seconds to read parameters from file".format(postreadTime - prereadTime))
				else: # No checkpoint exists
					print("Initializing model parameters")
					preInitTime = time.time()
					tf.global_variables_initializer().run()
					postInitTime = time.time()
					print("It took {} seconds to freshly initialize model parameters".format(postInitTime - preInitTime))

				# Train and validate
				iter_per_epoch = int(len(train_loader) / (args.single_batch_size*cfg.GPU_USE_COUNT))
				print('iter_per_epoch={}'.format(iter_per_epoch))
				is_summary, is_summary_image, is_validate = False, False, False
				save_model_interval = int(iter_per_epoch / 3)
				
				summary_interval = 5
				summary_image_interval = 20
				#summary_image_interval = 1
				save_model_interval = int(iter_per_epoch / 3)
				validate_interval = 60

				print('iter_per_epoch={}'.format(iter_per_epoch))
				print('summary_interval', summary_interval)
				print('summary_image_interval', summary_image_interval)
				print('save_model_interval', save_model_interval)
				print('validate_interval', validate_interval)

				summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
				startTraining = time.time()
				while model.epoch.eval() < args.max_epoch:
					is_summary, is_summary_image, is_validate = False, False, False
					iter = model.global_step.eval()
					print('iteration = {}'.format(iter))
					if not iter % summary_interval:
						is_summary = True
					if not iter % summary_image_interval:
						is_summary_image = True
					if not iter % save_model_interval:
						model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step)
					if not iter % validate_interval:
						is_validate = True
					if not iter % iter_per_epoch:
						sess.run(model.epoch_add_op)
						print('training epoch {} of {} total'.format(model.epoch.eval(), args.max_epoch))
					flag, data = train_loader.load(args.single_batch_size)
					if flag:
						train_loader.reset()
					ret = model.train_step(sess, data, train=True, summary=is_summary)
					print('train: {}/{} @ epoch:{}/{} loss: {} reg_loss: {} cls_loss: {} {}'.format(iter,
																									iter_per_epoch * args.max_epoch, 
																									model.epoch.eval(), args.max_epoch, 
																									ret[0], ret[1], ret[2], args.tag))
					print('Time since training started {} secs'.format(time.time() - startTraining))

					if is_summary:
						summary_writer.add_summary(ret[-1], iter)

					if is_summary_image:
						flag, valdat = valid_loader.load(args.single_batch_size)
						if flag:
							valid_loader.reset()
						ret = model.predict_step(sess, valdat, summary=True)
						summary_writer.add_summary(ret[-1], iter)

					if is_validate:
						flag, valdat = valid_loader.load(args.single_batch_size)
						if flag:
							valid_loader.reset()
						ret = model.validate_step(sess, valdat, summary=True)

					if check_if_should_pause(args.tag):
						model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step)

				stopTraining = time.time()
				print('Training took a total of {} secs'.format(stopTraining - startTraining))

if __name__ == '__main__':
	tf.app.run(main)

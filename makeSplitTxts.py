#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# makeSplitTxts.py
# Creates two text files for splitting training data into validation and test sets
# by G. Dylan Dickerson
# created 18 Dec 2019

import argparse
import numpy as np
from random import sample
import os

from config import cfg

if __name__ == "__main__":
	defaultDir = os.path.join(os.getcwd(), 'DataSplits')
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--trainNum', type=int, default=7000,
						help='number of members to keep in the training set')
	parser.add_argument('-d', '--splitDir', type=str, default=defaultDir,
						help='directory to save output files in')
	args = parser.parse_args()

	if args.trainNum > cfg.TRAIN_DATA_LEN:
		raise ValueError('The number of samples to be kept must be less than or equal to the number of training samples')

	fullList = np.arange(0, cfg.TRAIN_DATA_LEN+1)
	trainList = np.sort(np.random.choice(fullList, args.trainNum, replace=False))
	valList = np.setdiff1d(fullList, trainList)
	
	os.makedirs(args.splitDir, exist_ok=True)
	trainFile = os.path.join(args.splitDir, 'train.txt')
	valFile = os.path.join(args.splitDir, 'val.txt')

	with open(trainFile, 'w') as file1:
		for i in trainList:
			file1.write('{:06d}\n'.format(i))

	with open(valFile, 'w') as file2:
		for i in valList:
			file2.write('{:06d}\n'.format(i))

	print('There are {} members in the training set and {} in the validation set'.format(len(trainList), len(valList)))
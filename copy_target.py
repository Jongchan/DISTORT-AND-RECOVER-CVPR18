# coding: utf-8
from scipy.misc import imread
import skimage.color as color
import glob,os
import numpy as np
import shutil, os

target_indices=[3973, 3979, 4113, 4162, 4393, 4831, 17, 502, 816, 1052, 1415, 1580, 1793, 2049, 2495, 2680, 2867, 3012, 3252, 3322]
target_dir = '/data1/jcpark/Supervised_results/'
dirs = glob.glob('./*/step_*')
dirs = sorted(dirs)
for test_dir in dirs:
	print test_dir
	target_dir_1 = os.path.join(target_dir, test_dir.split('/')[1])
	if not os.path.exists(target_dir_1):
		os.mkdir(target_dir_1)
	target_dir_2 = os.path.join(target_dir, test_dir)
	os.mkdir(target_dir_2)
	for target_idx in target_indices:
		retouched_path = glob.glob(os.path.join(test_dir,'raw/%06d.jpg/*retouched*'%target_idx))
		if len(retouched_path)==1:
			shutil.copy2(retouched_path[0], target_dir_2)

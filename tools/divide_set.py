import glob,shutil,os
from random import shuffle


data_dir = '/hdd2/DATA/VA_crawler/shutterstock/cropped_resized'

train_dir	= os.path.join(data_dir, 'train')
test_dir	= os.path.join(data_dir, 'test')

train_raw_dir		= os.path.join(train_dir, 'raw')
train_target_dir	= os.path.join(train_dir, 'target')

test_raw_dir		= os.path.join(test_dir, 'raw')
test_target_dir		= os.path.join(test_dir, 'target')

image_paths	= glob.glob(os.path.join(train_target_dir, '*.jpg'))
shuffle(image_paths)
test_set	= image_paths[:1000]
train_set	= image_paths[len(test_set):]

for idx, train_target_path in enumerate(test_set):
	train_raw_paths = glob.glob(os.path.join(train_raw_dir, os.path.basename(train_target_path)+"*.jpg"))
	"""
	print len(train_raw_paths)
	if len(train_raw_paths)>1:
		print train_target_path
		print train_raw_paths
	"""
	for train_raw_path in train_raw_paths:
		if os.path.exists(train_raw_path):
			shutil.move(train_raw_path, test_raw_dir)
	if os.path.exists(train_target_path):
		shutil.move(train_target_path, test_target_dir)

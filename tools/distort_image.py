import os, glob, random, sys, math
from scipy.misc import imread,imresize
from PIL import Image
import numpy as np
from random import shuffle
import skimage.color as color
from non_linear import *

import argparse
parser = argparse.ArgumentParser(description="specify the start_idx and end_idx")
parser.add_argument('--start_idx', type=int)
parser.add_argument('--end_idx', type=int)
args = parser.parse_args()
start_idx = args.start_idx
end_idx = args.end_idx

dst_dir		= "/hdd2/jcpark/DATA/shutterstock/baseline_single"
img_dir		= "/hdd2/jcpark/DATA/shutterstock/baseline_single/distorted_test"
label_dir	= "/hdd2/jcpark/DATA/shutterstock/baseline_single/labels"
if not os.path.exists(dst_dir):
	os.mkdir(dst_dir)
if not os.path.exists(img_dir):
	os.mkdir(img_dir)
if not os.path.exists(label_dir):
	os.mkdir(label_dir)

raw_img_list = glob.glob("/hdd2/jcpark/DATA/shutterstock/cropped_resized/test/target/*.jpg")

if not (end_idx > start_idx and start_idx < len(raw_img_list)):
	print "index ain't right."
	sys.exit(1)
end_idx = min(end_idx, len(raw_img_list))

raw_img_list = sorted(raw_img_list)[start_idx:end_idx]

action_list = 	[
					brightness,
					color_saturation,
					contrast,
					hl_brightness,
					shadow_brightness,
					hl_saturation,
					shadow_saturation,
					red_adjust_red,
					green_adjust_green,
					blue_adjust_blue,
					cyan_adjust_cyan,
					magenta_adjust_magenta,
					yellow_adjust_yellow
				]


def take_action(image_np, idx, degree):

	return_np = action_list[idx](image_np+0.5, degree)
	return return_np-0.5

def distort_image(raw_image, image_path, lower_b, higher_b, distort_single=False, use_threshold=True):
	basename = os.path.basename(img_path)
	fn, ext = os.path.splitext(basename)

	actions = []
	# MULTIPLE 
	# SINGLE
	if distort_single:
		random_idx = random.randint(0, len(action_list)-1)
		sign = (random.random()<0.5)*2-1
		degree = random.uniform(0.1, 0.3)
		actions.append((random_idx, 1+degree*sign))
	else:
		for i in range(len(action_list)):
			if random.random() > 0.5:#apply random distortion from 0.3~1.0
				sign = (random.random()<0.5)*2-1
				degree = random.uniform(0.1, 0.2)
				actions.append((i, 1+degree*sign))

	shuffle(actions)
	image = raw_image.copy()
	#action_str = ""
	for action_pair in actions:
		idx, degree = action_pair
		image = take_action(image, idx, degree)
		#action_str += "%d_%.2f_" % (idx, float(degree))
	raw_image_lab = color.rgb2lab(raw_image+0.5)
	image_lab = color.rgb2lab(image+0.5)
	#mse = (( image_lab - raw_image_lab )**2).mean()/100
	mse = np.sqrt(np.sum(( raw_image_lab - image_lab)**2, axis=2)).mean()/10.0

	if use_threshold:
		if mse > lower_b and mse < higher_b:
			return image, True, mse, actions
		else:
			return None, False, mse, actions
	else:
		return image, True, mse, actions


for i, img_path in enumerate(raw_img_list):
	if i%100 == 0:
		print ("processed %d out of %d" %( i, len(raw_img_list) ))
	try:
		image = imresize(imread(img_path), (224,224))/255.0-0.5
		for j in range(50):
			#distorted_image, done, mse, actions = distort_image(image, img_path, 1.0, 2.0, use_threshold=True)
			distorted_image, done, mse, actions = distort_image(image, img_path, 1.0, 2.0, distort_single=True, use_threshold=False)
			if done:
				#raw = Image.fromarray(np.uint8(np.clip((image+0.5)*255, 0, 255)))
				#raw.save('./raw_images/%s'%(os.path.basename(img_path)))
				distorted = Image.fromarray(np.uint8(np.clip((distorted_image+0.5)*255, 0, 255)))
				distorted.save(os.path.join(img_dir,'%s__%f.jpg'%(os.path.basename(img_path), mse)))
				actions_np = np.array(actions)
				with open(os.path.join(label_dir, '%s__%f.jpg.label'%(os.path.basename(img_path), mse)), 'wb') as f:
					np.save(f, actions_np)
				break
	except Exception as e:
		print str(e)
	

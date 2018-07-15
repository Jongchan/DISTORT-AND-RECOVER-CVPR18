"""
2016-10-15

MIT5K dataset augmentation by cropping for reward network's finetuning

"""
import os, glob, random, sys
from scipy.misc import imread
from PIL import Image as im
sys.path.append(os.path.abspath('../reinforcement_learning/'))
from action import take_action


target_size = (224,224)

#save_dir = "./test/"

import argparse
parser = argparse.ArgumentParser(description="specify the location of input / output directory")
parser.add_argument('--input_dir', type=str)
parser.add_argument('--save_dir', type=str)
args = parser.parse_args()
input_dir = args.input_dir
save_dir = args.save_dir
if not (input_dir and save_dir):
	print "please specify --input_dir and --save_dir"
	sys.exit()
else:
	print "input_dir", input_dir
	print "save_dir", save_dir


#save_dir = "/hdd1/MIT5K/images/aug/0/"
raw_img_list = glob.glob(os.path.join(input_dir,"*.jpg"))

if not os.path.exists(save_dir):
	os.mkdir(save_dir)
	
def aug(image_path):

	basename = os.path.basename(image_path)
	name, ext = os.path.splitext(basename)

	img = im.open(image_path)
	w,h = img.size
	side = min(w,h)

	side_x = 224
	side_y = 224
	stride_x = 30
	stride_y = 30

	for i in range((w-side_x)//stride_x+1): # i for horizontal direction
		for j in range((h-side_y)//stride_y+1): # j for vertical direction
			cropped = img.crop((stride_x*i, stride_y*j, stride_x*i+side_x, stride_y*j+side_y))
			resized = cropped.resize(target_size, im.ANTIALIAS)
			resized.save(os.path.join(save_dir, name+"_"+str(i)+"_"+str(j)+ext))
			


for i, img_path in enumerate(raw_img_list):
	aug(img_path)
	if i%100 == 0:
		print ("processed %d out of %d" %( i, len(raw_img_list) ))

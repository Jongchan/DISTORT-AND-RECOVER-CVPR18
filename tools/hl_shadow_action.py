from scipy.misc import imread
import numpy as np
import glob
import skimage.color as color
import math
from PIL import Image
import os

#image_paths = glob.glob('./test_images/*.JPG')
#image_paths = glob.glob('./test_images/*_done.jpg')
image_paths = glob.glob('./test_images/*10_done.jpg')
"""
def f(x):
	#return min(100, math.tan(x/100*math.pi/4)*100/2)
	return (x/100.0)^0.4*100
f = np.vectorize(f)
"""
def f(x):
	return np.power((x/100.0), 0.8)*100
def curve_sigmoid(x):
	# sigmoid
	x = x/100.0
	width = 10

	top = 1/(1+math.exp(-width*0.5))
	bottom = 1/(1+math.exp(-width*(-0.5)))
	

	return (np.divide(1,(1+np.exp(-width*(x-0.5))))-bottom)/(top-bottom)

def curve_inv_sig(x):
	x = x/100.0
	x = np.clip(x, 0.001, 0.999)
	width = 10
	return np.log( np.divide( x, (1-x) ) ) / width + 0.5
	
def curve_1(x):
	mid = 0.5
	left_curvature = 0.8
	right_curvature = 1.6
	x = x/100.0
	return np.clip(np.power((x/mid), left_curvature)*mid, 0, 0.5) + np.power(np.clip((x-mid)/(1-mid), 0, 1), right_curvature)*(1-mid)
if True:
	image_rgb = imread(image_paths[0]).astype('float')/255.0
	image_lab = color.rgb2lab(image_rgb)
	image_l = image_lab[:,:,0]
	print image_l.max()
	print image_l
	#image_l_f = f(image_l)
	#image_l_f = curve_sigmoid(image_l)
	image_l_f = curve_inv_sig(image_l)
	print image_l_f
	image_lab[:,:,0] = image_l_f*100.0
	print image_lab[:,:,0]
	image_rgb_f = color.lab2rgb(image_lab)*255.0
	image_f = Image.fromarray(image_rgb_f.astype('uint8'), 'RGB')
	image_f.save('./test_images/'+os.path.basename(image_paths[0])+"10_done.jpg")

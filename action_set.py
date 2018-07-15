from scipy.misc import imread, imresize
import numpy as np
import glob
import skimage.color as color
import math
from PIL import Image
import os
def B_sigmoid_low(rgb, width):
	l = rgb[:,:,2]
	l_final = sigmoid_low(l,width)
	ret  = rgb.copy()
	ret[:,:,2] = l_final
	return ret

def B_sigmoid_high(rgb, width):
	l = rgb[:,:,2]
	l_final = sigmoid_high(l,width)
	ret  = rgb.copy()
	ret[:,:,2] = l_final
	return ret

def B_inv_sigmoid_low(rgb, width):
	l = rgb[:,:,2]
	l_final = inv_sigmoid_low(l,width)
	ret  = rgb.copy()
	ret[:,:,2] = l_final
	return ret

def B_inv_sigmoid_high(rgb, width):
	l = rgb[:,:,2]
	l_final = inv_sigmoid_high(l,width)
	ret  = rgb.copy()
	ret[:,:,2] = l_final
	return ret


def G_sigmoid_low(rgb, width):
	l = rgb[:,:,1]
	l_final = sigmoid_low(l,width)
	ret  = rgb.copy()
	ret[:,:,1] = l_final
	return ret

def G_sigmoid_high(rgb, width):
	l = rgb[:,:,1]
	l_final = sigmoid_high(l,width)
	ret  = rgb.copy()
	ret[:,:,1] = l_final
	return ret

def G_inv_sigmoid_low(rgb, width):
	l = rgb[:,:,1]
	l_final = inv_sigmoid_low(l,width)
	ret  = rgb.copy()
	ret[:,:,1] = l_final
	return ret

def G_inv_sigmoid_high(rgb, width):
	l = rgb[:,:,1]
	l_final = inv_sigmoid_high(l,width)
	ret  = rgb.copy()
	ret[:,:,1] = l_final
	return ret

def R_sigmoid_low(rgb, width):
	l = rgb[:,:,0]
	l_final = sigmoid_low(l,width)
	ret  = rgb.copy()
	ret[:,:,0] = l_final
	return ret

def R_sigmoid_high(rgb, width):
	l = rgb[:,:,0]
	l_final = sigmoid_high(l,width)
	ret  = rgb.copy()
	ret[:,:,0] = l_final
	return ret

def R_inv_sigmoid_low(rgb, width):
	l = rgb[:,:,0]
	l_final = inv_sigmoid_low(l,width)
	ret  = rgb.copy()
	ret[:,:,0] = l_final
	return ret

def R_inv_sigmoid_high(rgb, width):
	l = rgb[:,:,0]
	l_final = inv_sigmoid_high(l,width)
	ret  = rgb.copy()
	ret[:,:,0] = l_final
	return ret

def L_sigmoid_low(lab, width):
	l = lab[:,:,0]
	l=l/100.0
	l_final = sigmoid_low(l,width)
	lab[:,:,0] = l_final*100.0
	return lab

def L_sigmoid_high(lab, width):
	l = lab[:,:,0]
	l=l/100.0
	l_final = sigmoid_high(l,width)
	lab[:,:,0] = l_final*100.0
	return lab

def L_inv_sigmoid_low(lab, width):
	l = lab[:,:,0]
	l=l/100.0
	l_final = inv_sigmoid_low(l,width)
	lab[:,:,0] = l_final*100.0
	return lab

def L_inv_sigmoid_high(lab, width):
	l = lab[:,:,0]
	l=l/100.0
	l_final = inv_sigmoid_high(l,width)
	lab[:,:,0] = l_final*100.0
	return lab


def sigmoid_low(in_channel, width):
	#l = lab[:,:,0]
	#l=l/100.0

	top = 1/(1+math.exp(-width*0.375))
	bottom = 1/(1+math.exp(-width*-0.625))
	"""
	in_channel = in_channel*(top-bottom)+bottom
	in_channel = np.clip(in_channel, 0.000001, 0.999999)
	"""
	#in_final = np.log( np.divide( in_channel, (1-in_channel) ) ) / width + 0.625
	in_final = (np.divide(1, (1+np.exp(-width*(in_channel-0.625))))-bottom)/(top-bottom)
	#lab[:,:,0] = l_final*100.0
	return in_final

def sigmoid_high(in_channel, width):
	#l = lab[:,:,0]
	#l=l/100.0

	top = 1/(1+math.exp(-width*0.625))
	bottom = 1/(1+math.exp(-width*-0.375))
	"""
	print 'top', top
	print 'bottom', bottom
	print 'in_channel', in_channel
	#in_channel = in_channel*(top-bottom)+bottom
	print 'in_channel', in_channel
	#in_channel = np.clip(in_channel, 0.000001, 0.999999)
	"""
	#in_final = np.log( np.divide( in_channel, (1-in_channel) ) ) / width + 0.625
	in_final = (np.divide(1, (1+np.exp(-width*(in_channel-0.375))))-bottom)/(top-bottom)
	#lab[:,:,0] = l_final*100.0
	return in_final

def inv_sigmoid_low(in_channel, width):
	#l = lab[:,:,0]
	#l=l/100.0

	top = 1/(1+math.exp(-width*0.375))
	bottom = 1/(1+math.exp(-width*-0.625))
	in_channel = in_channel*(top-bottom)+bottom
	in_channel = np.clip(in_channel, 0.000001, 0.999999)
	in_final = np.log( np.divide( in_channel, (1-in_channel) ) ) / width + 0.625
	#l_final = (np.divide(1, (1+np.exp(-width*(l-0.5))))-bottom)/(top-bottom)
	#lab[:,:,0] = l_final*100.0
	return in_final

def inv_sigmoid_high(in_channel, width):
	#l = lab[:,:,0]
	#l=l/100.0

	top = 1/(1+math.exp(-width*0.625))
	bottom = 1/(1+math.exp(-width*-0.375))
	in_channel = in_channel*(top-bottom)+bottom
	in_channel = np.clip(in_channel, 0.000001, 0.999999)
	in_final = np.log( np.divide( in_channel, (1-in_channel) ) ) / width + 0.375
	#l_final = (np.divide(1, (1+np.exp(-width*(l-0.5))))-bottom)/(top-bottom)
	#lab[:,:,0] = l_final*100.0
	return in_final


"""
def inv_sigmoid_low(lab, width):
	l = lab[:,:,0]
	l=l/100.0
	top = 1/(1+math.exp(-width*0.375))
	bottom = 1/(1+math.exp(-width*-0.625))
	l = l*(top-bottom)+bottom
	l = np.clip(l, 0.000001, 0.999999)
	l_final = np.log( np.divide( l, (1-l) ) ) / width + 0.625
	#l_final = (np.divide(1, (1+np.exp(-width*(l-0.5))))-bottom)/(top-bottom)
	lab[:,:,0] = l_final*100.0
	return lab

def inv_sigmoid_high(lab, width):
	l = lab[:,:,0]
	l=l/100.0
	top = 1/(1+math.exp(-width*0.625))
	bottom = 1/(1+math.exp(-width*-0.375))
	l = l*(top-bottom)+bottom
	l = np.clip(l, 0.000001, 0.999999)
	l_final = np.log( np.divide( l, (1-l) ) ) / width + 0.375
	#l_final = (np.divide(1, (1+np.exp(-width*(l-0.5))))-bottom)/(top-bottom)
	lab[:,:,0] = l_final*100.0
	return lab
"""

def contrast(image_rgb, b):
	mean = np.mean(image_rgb)
	degenerate = np.zeros(image_rgb.shape)+mean
	
	image_rgb = b * image_rgb + (1-b) * degenerate
	image_rgb = np.clip(image_rgb,0,1)
	return image_rgb

def brightness(image_rgb, b):
	degenerate = np.zeros(image_rgb.shape)
	
	image_rgb = b * image_rgb + (1-b) * degenerate
	image_rgb = np.clip(image_rgb,0,1)
	return image_rgb

def color_saturation(image_rgb, b):
	degenerate = image_rgb.mean(axis=2)

	image_rgb[:,:,0] = b * image_rgb[:,:,0] + (1-b) * degenerate
	image_rgb[:,:,1] = b * image_rgb[:,:,1] + (1-b) * degenerate
	image_rgb[:,:,2] = b * image_rgb[:,:,2] + (1-b) * degenerate
	image_rgb = np.clip(image_rgb,0,1)
	return image_rgb

def white_bal(image_rgb, r,g,b):
	image_rgb[:,:,0] = r/255.0 * image_rgb[:,:,0]
	image_rgb[:,:,1] = g/255.0 * image_rgb[:,:,1]
	image_rgb[:,:,2] = b/255.0 * image_rgb[:,:,2]
	image_rgb = np.clip(image_rgb,0,1)
	return image_rgb
	

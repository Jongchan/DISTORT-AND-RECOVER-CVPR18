import os, glob, math, sys
import skimage.color as color
from scipy.misc import imread
import numpy as np
from PIL import Image
from convert import *


def contrast(image_rgb, b, w=None):
	mean = np.mean(image_rgb)
	degenerate = np.zeros(image_rgb.shape)+mean
	if w is None:
		w = np.ones(degenerate[:,:,0].shape)
	
	weighted_diff = (1-b) * ( degenerate - image_rgb )
	weighted_diff[:,:,0] = weighted_diff[:,:,0] * w 
	weighted_diff[:,:,1] = weighted_diff[:,:,1] * w 
	weighted_diff[:,:,2] = weighted_diff[:,:,2] * w 

	image_rgb = weighted_diff + image_rgb
	image_rgb = np.clip(image_rgb,0,1)
	return image_rgb

def brightness(image_rgb, b, w=None):
	degenerate = np.zeros(image_rgb.shape)
	if w is None:
		w = np.ones(degenerate[:,:,0].shape)
	
	weighted_diff = (1-b) * ( degenerate - image_rgb )
	weighted_diff[:,:,0] = weighted_diff[:,:,0] * w 
	weighted_diff[:,:,1] = weighted_diff[:,:,1] * w 
	weighted_diff[:,:,2] = weighted_diff[:,:,2] * w 
	
	image_rgb = weighted_diff + image_rgb
	image_rgb = np.clip(image_rgb,0,1)
	return image_rgb

def color_saturation(image_rgb, b, w=None):
	degenerate = image_rgb.mean(axis=2)
	if w is None:
		w = np.ones(degenerate.shape)
	image_rgb[:,:,0] += (1-b) * (degenerate - image_rgb[:,:,0]) * w
	image_rgb[:,:,1] += (1-b) * (degenerate - image_rgb[:,:,1]) * w
	image_rgb[:,:,2] += (1-b) * (degenerate - image_rgb[:,:,2]) * w
	image_rgb = np.clip(image_rgb,0,1)
	return image_rgb

def get_shadow_weight_filter(x, shadow_threshold=0.3, steepness=15):
	return 1 - ( np.tanh( ( x - shadow_threshold ) * steepness) + 1 ) / 2

def get_hl_weight_filter(x, hl_threshold=0.7, steepness=15):
	return ( np.tanh( ( x - hl_threshold ) * steepness) + 1 ) / 2

def hl_brightness(rgb, degree):
	lab = color.rgb2lab(rgb)
	l = lab[:,:,0]/100.0
	w = get_hl_weight_filter(l)
	return brightness(rgb, degree, w=w)

def shadow_brightness(rgb, degree):
	lab = color.rgb2lab(rgb)
	l = lab[:,:,0]/100.0
	w = get_shadow_weight_filter(l)
	return brightness(rgb, degree, w=w)

def hl_saturation(rgb, degree):
	lab = color.rgb2lab(rgb)
	l = lab[:,:,0]/100.0
	w = get_hl_weight_filter(l)
	return color_saturation(rgb, degree, w=w)

def shadow_saturation(rgb, degree):
	lab = color.rgb2lab(rgb)
	l = lab[:,:,0]/100.0
	w = get_shadow_weight_filter(l)
	return color_saturation(rgb, degree, w=w)

def hl_contrast(rgb, degree):
	lab = color.rgb2lab(rgb)
	l = lab[:,:,0]/100.0
	w = get_hl_weight_filter(l)
	return contrast(rgb, degree, w=w)

def shadow_contrast(rgb, degree):
	lab = color.rgb2lab(rgb)
	l = lab[:,:,0]/100.0
	w = get_shadow_weight_filter(l)
	return contrast(rgb, degree, w=w)

def cyan_adjust_cyan(rgb, degree):
	cmyk = rgb2cmyk(rgb)
	w = get_hl_weight_filter(cmyk[:,:,0], 0.5)

	degenerate = np.zeros(w.shape)
	weighted_diff = (1-degree) * ( degenerate - cmyk[:,:,0] )
	weighted_diff[:,:] = weighted_diff[:,:] * w 
	cmyk[:,:,0] += weighted_diff
	cmyk = np.clip(cmyk, 0, 1)
	return cmyk2rgb(cmyk)


def magenta_adjust_magenta(rgb, degree):
	cmyk = rgb2cmyk(rgb)
	w = get_hl_weight_filter(cmyk[:,:,1], 0.5)

	degenerate = np.zeros(w.shape)
	weighted_diff = (1-degree) * ( degenerate - cmyk[:,:,1] )
	weighted_diff[:,:] = weighted_diff[:,:] * w 
	cmyk[:,:,1] += weighted_diff
	cmyk = np.clip(cmyk, 0, 1)
	return cmyk2rgb(cmyk)


def yellow_adjust_yellow(rgb, degree):
	cmyk = rgb2cmyk(rgb)
	w = get_hl_weight_filter(cmyk[:,:,2], 0.5)

	degenerate = np.zeros(w.shape)
	weighted_diff = (1-degree) * ( degenerate - cmyk[:,:,2] )
	weighted_diff[:,:] = weighted_diff[:,:] * w 
	cmyk[:,:,2] += weighted_diff
	cmyk = np.clip(cmyk, 0, 1)
	return cmyk2rgb(cmyk)


def red_adjust_red(rgb, degree):
	w = get_hl_weight_filter(rgb[:,:,0], 0.7)

	degenerate = np.zeros(w.shape)
	weighted_diff = (1-degree) * ( degenerate - rgb[:,:,0] )
	weighted_diff[:,:] = weighted_diff[:,:] * w 
	rgb[:,:,0] += weighted_diff
	rgb = np.clip(rgb, 0, 1)
	return rgb


def green_adjust_green(rgb, degree):
	w = get_hl_weight_filter(rgb[:,:,0], 0.7)

	degenerate = np.zeros(w.shape)
	weighted_diff = (1-degree) * ( degenerate - rgb[:,:,1] )
	weighted_diff[:,:] = weighted_diff[:,:] * w 
	rgb[:,:,1] += weighted_diff
	rgb = np.clip(rgb, 0, 1)
	return rgb


def blue_adjust_blue(rgb, degree):
	w = get_hl_weight_filter(rgb[:,:,0], 0.7)

	degenerate = np.zeros(w.shape)
	weighted_diff = (1-degree) * ( degenerate - rgb[:,:,2] )
	weighted_diff[:,:] = weighted_diff[:,:] * w 
	rgb[:,:,2] += weighted_diff
	rgb = np.clip(rgb, 0, 1)
	return rgb



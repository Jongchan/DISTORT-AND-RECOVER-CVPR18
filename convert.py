import numpy as np
from scipy.misc import imread
import os,glob

def rgb2cmyk(rgb):
	max_rgb = np.clip(np.amax(rgb,axis=2), 0.000001,1)
	K = 1 - max_rgb
	C = np.divide((max_rgb-rgb[:,:,0]), max_rgb)
	M = np.divide((max_rgb-rgb[:,:,1]), max_rgb)
	Y = np.divide((max_rgb-rgb[:,:,2]), max_rgb)
	CMYK = np.stack([C,M,Y,K], axis=-1)
	return CMYK

def cmyk2rgb(cmyk):
	R = np.multiply((1 - cmyk[:,:,0]), (1 - cmyk[:,:,-1]))
	G = np.multiply((1 - cmyk[:,:,1]), (1 - cmyk[:,:,-1]))
	B = np.multiply((1 - cmyk[:,:,2]), (1 - cmyk[:,:,-1]))
	RGB = np.stack([R,G,B], axis=-1)
	return RGB

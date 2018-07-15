import numpy as np
import skimage.color as color
from scipy.misc import imread, imresize
from PIL import Image
from sklearn.preprocessing import normalize
"distorted.jpg"
"raw.jpg"
# all image input is RGB whitened to (-0.5, 0.5)
# return the flattened histogram
def rgbl_hist(image_data):
	r_h, r_b = np.histogram(image_data[:,:,0], 256, [-0.5,0.5])
	g_h, g_b = np.histogram(image_data[:,:,1], 256, [-0.5,0.5])
	b_h, b_b = np.histogram(image_data[:,:,2], 256, [-0.5,0.5])
	l = image_data[:,:,0]*0.2126 + image_data[:,:,1]*0.7152 + image_data[:,:,2]*0.0722
	l_h, l_b = np.histogram(l, 256, [-0.5,0.5])
	return np.concatenate((r_h, g_h, b_h, l_h), axis=0)/1000.0

def tiny_hist_old(image_np):
	image_np_rgb = image_np + 0.5 # 0~1 rgb
	image_np_lab = color.rgb2lab(image_np_rgb)
	
	image_np_lab_tiny = imresize(image_np_lab, (16,16), interp='bicubic')

	return image_np_lab_tiny.reshape(16*16*3)/100.0

def tiny_hist(image_np):
	image_pil = Image.fromarray(np.uint8((image_np+0.5)*255))
	image_pil.thumbnail((16,16),Image.ANTIALIAS)
	image_np_resized = np.asarray(image_pil)/255.0
	image_np_lab = color.rgb2lab(image_np_resized)

	return image_np_lab.reshape(16*16*3)/100.0
def tiny_hist_28(image_np):
	image_pil = Image.fromarray(np.uint8((image_np+0.5)*255))
	image_pil.thumbnail((28,28),Image.ANTIALIAS)
	image_np_resized = np.asarray(image_pil)/255.0
	image_np_lab = color.rgb2lab(image_np_resized)

	return image_np_lab.reshape(28*28*3)/100.0

def lab_hist(image_np):
	image_np_rgb = image_np + 0.5 # 0~1 rgb
	image_np_lab = color.rgb2lab(image_np_rgb)

	num_bin_L = 10
	num_bin_a = 10
	num_bin_b = 10

	L_max = 100
	L_min = 0
	a_max = 60
	a_min = -60
	b_max = 60
	b_min = -60
	image_np_lab = image_np_lab.reshape([224*224,3])
	H, edges = np.histogramdd(image_np_lab, bins=(num_bin_L, num_bin_a, num_bin_b), \
			range=((L_min, L_max), (a_min, a_max), (b_min, b_max)))
	return H.reshape(1000)/1000.0

def lab_hist_8k(image_np):
	image_np_rgb = image_np + 0.5 # 0~1 rgb
	image_np_lab = color.rgb2lab(image_np_rgb)

	num_bin_L = 20
	num_bin_a = 20
	num_bin_b = 20

	L_max = 100
	L_min = 0
	a_max = 60
	a_min = -60
	b_max = 60
	b_min = -60
	image_np_lab = image_np_lab.reshape([224*224,3])
	H, edges = np.histogramdd(image_np_lab, bins=(num_bin_L, num_bin_a, num_bin_b), \
			range=((L_min, L_max), (a_min, a_max), (b_min, b_max)))

	#return H.reshape(8000)/100.0
	return normalize(H.reshape(1,8000)).ravel()

#return image_np_lab
def example(path):
	img_example = imread(path).astype('float')/255.0-0.5

	return tiny_hist(img_example)

from PIL import Image, ImageEnhance
import skimage.color as color
import numpy as np
import time
action_size = 12
import math
from action_set import *
#from action_set_tf import *
def PIL2array(img):
	return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

def convert_K_to_RGB(colour_temperature):
    """
    Converts from K to RGB, algorithm courtesy of 
    http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    """
    #range check
    if colour_temperature < 1000: 
        colour_temperature = 1000
    elif colour_temperature > 40000:
        colour_temperature = 40000
    
    tmp_internal = colour_temperature / 100.0
    
    # red 
    if tmp_internal <= 66:
        red = 255
    else:
        tmp_red = 329.698727446 * math.pow(tmp_internal - 60, -0.1332047592)
        if tmp_red < 0:
            red = 0
        elif tmp_red > 255:
            red = 255
        else:
            red = tmp_red
    
    # green
    if tmp_internal <=66:
        tmp_green = 99.4708025861 * math.log(tmp_internal) - 161.1195681661
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green
    else:
        tmp_green = 288.1221695283 * math.pow(tmp_internal - 60, -0.0755148492)
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green
    
    # blue
    if tmp_internal >=66:
        blue = 255
    elif tmp_internal <= 19:
        blue = 0
    else:
        tmp_blue = 138.5177312231 * math.log(tmp_internal - 10) - 305.0447927307
        if tmp_blue < 0:
            blue = 0
        elif tmp_blue > 255:
            blue = 255
        else:
            blue = tmp_blue
    
    return red, green, blue

"""
def take_action(image_np, action_idx, sess):
	#image_pil = Image.fromarray(np.uint8(image_np))
	#image_pil = Image.fromarray(np.uint8((image_np+0.5)*255))
	# enhance contrast
	return_np = None
	if action_idx == 0:
		return_np = contrast(image_np+0.5, 0.95,sess)
	elif action_idx == 1:
		return_np = contrast(image_np+0.5, 1.05,sess)
	# enhance color
	elif action_idx == 2:
		return_np = color_saturation(image_np+0.5, 0.95,sess)
	elif action_idx == 3:
		return_np = color_saturation(image_np+0.5, 1.05,sess)
	# color brightness
	elif action_idx == 4:
		return_np = brightness(image_np+0.5, 0.93,sess)
	elif action_idx == 5:
		return_np = brightness(image_np+0.5, 1.07,sess)
	# color temperature : http://stackoverflow.com/questions/11884544/setting-color-temperature-for-a-given-image-like-in-photoshop
	elif action_idx == 6:
		r,g,b = 240, 240, 255 # around 6300K #convert_K_to_RGB(6000)
		return_np = white_bal(image_np+0.5, r,g,b,sess)
	elif action_idx == 7:
		r,g,b = 270, 270, 255 # around 6300K #convert_K_to_RGB(6000)
		return_np = white_bal(image_np+0.5, r,g,b,sess)
	elif action_idx == 8:
		r,g,b = 255, 240, 240 # around 6300K #convert_K_to_RGB(6000)
		return_np = white_bal(image_np+0.5, r,g,b,sess)
	elif action_idx == 9:
		r,g,b = 255, 270, 270 # around 6300K #convert_K_to_RGB(6000)
		return_np = white_bal(image_np+0.5, r,g,b,sess)
	elif action_idx == 10:
		r,g,b = 240, 255, 240 # around 6300K #convert_K_to_RGB(6000)
		return_np = white_bal(image_np+0.5, r,g,b,sess)
	elif action_idx == 11:
		r,g,b = 270, 255, 270 # around 6300K #convert_K_to_RGB(6000)
		return_np = white_bal(image_np+0.5, r,g,b,sess)
	return return_np-0.5
"""
def take_action(image_np, action_idx):
	#image_pil = Image.fromarray(np.uint8(image_np))
	#image_pil = Image.fromarray(np.uint8((image_np+0.5)*255))
	# enhance contrast
	return_np = None
	if action_idx == 0:
		return_np = contrast(image_np+0.5, 0.95)
	elif action_idx == 1:
		return_np = contrast(image_np+0.5, 1.05)
	# enhance color
	elif action_idx == 2:
		return_np = color_saturation(image_np+0.5, 0.95)
	elif action_idx == 3:
		return_np = color_saturation(image_np+0.5, 1.05)
	# color brightness
	elif action_idx == 4:
		return_np = brightness(image_np+0.5, 0.93)
	elif action_idx == 5:
		return_np = brightness(image_np+0.5, 1.07)
	# color temperature : http://stackoverflow.com/questions/11884544/setting-color-temperature-for-a-given-image-like-in-photoshop
	elif action_idx == 6:
		r,g,b = 240, 240, 255 # around 6300K #convert_K_to_RGB(6000)
		return_np = white_bal(image_np+0.5, r,g,b)
	elif action_idx == 7:
		r,g,b = 270, 270, 255 # around 6300K #convert_K_to_RGB(6000)
		return_np = white_bal(image_np+0.5, r,g,b)
	elif action_idx == 8:
		r,g,b = 255, 240, 240 # around 6300K #convert_K_to_RGB(6000)
		return_np = white_bal(image_np+0.5, r,g,b)
	elif action_idx == 9:
		r,g,b = 255, 270, 270 # around 6300K #convert_K_to_RGB(6000)
		return_np = white_bal(image_np+0.5, r,g,b)
	elif action_idx == 10:
		r,g,b = 240, 255, 240 # around 6300K #convert_K_to_RGB(6000)
		return_np = white_bal(image_np+0.5, r,g,b)
	elif action_idx == 11:
		r,g,b = 270, 255, 270 # around 6300K #convert_K_to_RGB(6000)
		return_np = white_bal(image_np+0.5, r,g,b)
	elif action_idx==12:
		image_lab = color.rgb2lab(image_np+0.5)
		image_lab = L_sigmoid_low(image_lab, 4)
		return_np = color.lab2rgb(image_lab)
	elif action_idx==13:
		image_lab = color.rgb2lab(image_np+0.5)
		image_lab = L_sigmoid_high(image_lab, 4)
		return_np = color.lab2rgb(image_lab)
	elif action_idx==14:
		image_lab = color.rgb2lab(image_np+0.5)
		image_lab = L_inv_sigmoid_low(image_lab, 4)
		return_np = color.lab2rgb(image_lab)
	elif action_idx==15:
		image_lab = color.rgb2lab(image_np+0.5)
		image_lab = L_inv_sigmoid_high(image_lab, 4)
		return_np = color.lab2rgb(image_lab)
	elif action_idx==16:
		image_rgb = image_np+0.5
		return_np = R_sigmoid_low(image_rgb, 4)
	elif action_idx==17:
		image_rgb = image_np+0.5
		return_np = R_sigmoid_high(image_rgb, 4)
	elif action_idx==18:
		image_rgb = image_np+0.5
		return_np = R_inv_sigmoid_low(image_rgb, 4)
	elif action_idx==19:
		image_rgb = image_np+0.5
		return_np = R_inv_sigmoid_high(image_rgb, 4)
	elif action_idx==20:
		image_rgb = image_np+0.5
		return_np = G_sigmoid_low(image_rgb, 4)
	elif action_idx==21:
		image_rgb = image_np+0.5
		return_np = G_sigmoid_high(image_rgb, 4)
	elif action_idx==22:
		image_rgb = image_np+0.5
		return_np = G_inv_sigmoid_low(image_rgb, 4)
	elif action_idx==23:
		image_rgb = image_np+0.5
		return_np = G_inv_sigmoid_high(image_rgb, 4)
	elif action_idx==24:
		image_rgb = image_np+0.5
		return_np = B_sigmoid_low(image_rgb, 4)
	elif action_idx==25:
		image_rgb = image_np+0.5
		return_np = B_sigmoid_high(image_rgb, 4)
	elif action_idx==26:
		image_rgb = image_np+0.5
		return_np = B_inv_sigmoid_low(image_rgb, 4)
	elif action_idx==27:
		image_rgb = image_np+0.5
		return_np = B_inv_sigmoid_high(image_rgb, 4)
	else:
		print "error"
	return return_np-0.5

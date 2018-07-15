"""
2016-09-28

MIT5K dataset augmentation by applying predefined random actions
(original input data is too dark and low contrast in general.)
"""
import os, glob, random, sys
from scipy.misc import imread
from PIL import Image as im

raw_img_list = glob.glob("/hdd2/MIT5K/images/raw/3/*.jpg")
raw_img_list = sorted(raw_img_list)

save_dir = "/hdd2/MIT5K/images/aug/retoucher_3_mild_distortion/"
import math
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
def take_action(img, idx, degree):
	from PIL import ImageEnhance
	if idx==0: # contrast
		enh = ImageEnhance.Contrast(img)
		img_enh = enh.enhance(degree)
	elif idx==1:
		enh = ImageEnhance.Color(img)
		img_enh = enh.enhance(degree)
	elif idx==2:
		enh = ImageEnhance.Brightness(img)
		img_enh = enh.enhance(degree)
	elif idx==3:
		r,g,b = convert_K_to_RGB(degree)
 		matrix = ( 	r / 255.0, 0.0, 0.0, 0.0,
               		0.0, g / 255.0, 0.0, 0.0,
               		0.0, 0.0, b / 255.0, 0.0 )
		img_enh = img.convert('RGB', matrix)
	else:
		return None
	return img_enh

if not os.path.exists(save_dir):
	os.mkdir(save_dir)


def random_aug(image_path):
	image = im.open(image_path)
	basename = os.path.basename(img_path)
	fn, ext = os.path.splitext(basename)

	actions = []
	for i in range(4):
		if random.random() > 0.7:#apply random distortion from 0.3~1.0
			if i == 3:
				# apply color temperature changes..
				# there are 2 extremes for distortion (1000,6000) and (11000,15000)

				temp_lb_min = 5000 # lowest temp bound
				temp_lb_max = 5800 # max value for lower extreme
				temp_b = 6000 # boundary in the middle
				temp_ub_min = 6200 # min val for higher extreme 
				temp_ub_max = 7000 # max val for higher extreme
				color_temperature = random.randint(temp_lb_min, temp_ub_max)
				if color_temperature < temp_b:
					color_temperature = (float(color_temperature)-temp_lb_min)/(temp_b-temp_lb_min)*(temp_lb_max-temp_lb_min)+temp_lb_min
				else:
					color_temperature = temp_ub_max - (temp_ub_max - float(color_temperature))/(temp_ub_max-temp_b)*(temp_ub_max-temp_ub_min)
				actions.append((i, color_temperature))
				continue
			sign = (random.random()<0.5)*2-1
			degree = random.uniform(0.1, 0.2)
			actions.append((i, 1+degree*sign))
	if len(actions)<2:
		return False
	from random import shuffle
	shuffle(actions)

	action_str = ""
	for action_pair in actions:
		idx, degree = action_pair
		image = take_action(image, idx, degree)
		action_str += "%d_%.2f_" % (idx, float(degree))
	image.save(os.path.join(save_dir, basename+"__"+action_str+ext))
	return True
for r in xrange(0,10):
	for i, img_path in enumerate(raw_img_list):
		random_aug(img_path)
		if i%100 == 0:
			print ("processed %d out of %d" %( i+r*len(raw_img_list), 10*len(raw_img_list) ))

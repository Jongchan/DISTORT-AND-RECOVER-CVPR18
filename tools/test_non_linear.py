from non_linear import *

image_paths = glob.glob('./test2.jpg')


degree = 1.2

#def hl_brightness(rgb, degree):
image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = hl_brightness(image_rgb, degree)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./0_hl_brightness.jpg')

image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = shadow_brightness(image_rgb, degree)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./0_shadow_brightness.jpg')

image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = hl_saturation(image_rgb, degree)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./0_hl_saturation.jpg')

image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = shadow_saturation(image_rgb, degree)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./0_shadow_saturation.jpg')

image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = cyan_adjust_cyan(image_rgb, degree)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./0_cyan.jpg')

image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = magenta_adjust_magenta(image_rgb, degree)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./0_magenta.jpg')

image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = yellow_adjust_yellow(image_rgb, degree)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./0_yellow.jpg')

image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = red_adjust_red(image_rgb, degree)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./0_red.jpg')

image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = green_adjust_green(image_rgb, degree)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./0_green.jpg')

image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = blue_adjust_blue(image_rgb, degree)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./0_blue.jpg')

image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = brightness(image_rgb, degree)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./0_brightness.jpg')

image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = color_saturation(image_rgb, degree)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./0_saturation.jpg')

image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = contrast(image_rgb, degree)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./0_contrast.jpg')

sys.exit(1)




"""
image_distorted, w, cmyk = cyan_adjust_cyan(image_rgb, 1.5)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./cyan_enhanced.jpg')
image_w_pil = Image.fromarray( (w*255).astype(np.uint8))
image_w_pil.save('./cyan_w.jpg')
image_c_pil = Image.fromarray( (rgb2cmyk(image_rgb)[:,:,0]*255).astype(np.uint8))
image_c_pil.save('./cyan.jpg')

image_distorted, w, cmyk = magenta_adjust_magenta(image_rgb, 1.5)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./magenta_enhanced.jpg')
image_w_pil = Image.fromarray( (w*255).astype(np.uint8))
image_w_pil.save('./magenta_w.jpg')
image_c_pil = Image.fromarray( (rgb2cmyk(image_rgb)[:,:,1]*255).astype(np.uint8))
image_c_pil.save('./magenta.jpg')

image_distorted, w, cmyk = yellow_adjust_yellow(image_rgb, 1.5)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./yellow_enhanced.jpg')
image_w_pil = Image.fromarray( (w*255).astype(np.uint8))
image_w_pil.save('./yellow_w.jpg')
image_c_pil = Image.fromarray( (rgb2cmyk(image_rgb)[:,:,2]*255).astype(np.uint8))
image_c_pil.save('./yellow.jpg')
"""
image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = red_adjust_red(image_rgb, 1.1)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./red_enhanced.jpg')
image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = green_adjust_green(image_rgb, 1.1)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./green_enhanced.jpg')
image_rgb = imread(image_paths[0]).astype('float')/255.0
image_distorted = blue_adjust_blue(image_rgb, 1.1)
image_distorted_pil = Image.fromarray( (image_distorted*255).astype(np.uint8), "RGB")
image_distorted_pil.save('./blue_enhanced.jpg')
sys.exit(1)





image_lab = color.rgb2lab(image_rgb)
image_l = image_lab[:,:,0]
print "max", np.max(image_l)
image_hl_filter = get_hl_weight_filter(image_l/100)
image_shadow_filter = get_shadow_weight_filter(image_l/100)
"""
image_hl = Image.fromarray((image_hl_filter*255).astype(np.uint8))
image_shadow = Image.fromarray((image_shadow_filter*255).astype(np.uint8))
image_hl.save('./hl.jpg')
image_shadow.save('./shadow.jpg')
"""
hl_br = brightness(image_rgb, 0.7, image_hl_filter)
hl_br_image = Image.fromarray((hl_br*255).astype(np.uint8), "RGB")
hl_br_image.save('./hl_br_image.jpg')
hl_ct = contrast(image_rgb, 1.8, image_hl_filter)
hl_ct_image = Image.fromarray((hl_ct*255).astype(np.uint8), "RGB")
hl_ct_image.save('./hl_ct_image.jpg')
shadow_br = brightness(image_rgb, 1.5, image_shadow_filter)
shadow_br_image = Image.fromarray((shadow_br*255).astype(np.uint8), 'RGB')
shadow_br_image.save('./shadow_br_image.jpg')
shadow_ct = contrast(image_rgb, 1.5, image_shadow_filter)
shadow_ct_image = Image.fromarray((shadow_ct*255).astype(np.uint8), 'RGB')
shadow_ct_image.save('./shadow_ct_image.jpg')

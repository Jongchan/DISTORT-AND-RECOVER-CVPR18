# coding: utf-8
import os
from crawler import *

crawled_count = 0
crawl_dir = '/hdd2/DATA/VA_crawler/shutterstock/by_query/'
keyword = 'group'
save_dir = os.path.join(crawl_dir, keyword)
if not os.path.exists(save_dir):
	os.mkdir(save_dir)

for i in range (200):
	print "get page number %d..."%(i+1)
	try:
		bs = get_url('https://www.shutterstock.com/search?searchterm=%s&search_source=base_search_form&language=ko&page=%d&sort=popular&image_type=photo&safe=true'%(keyword, i+1))
		img_wraps=bs.findAll('div', attrs={'class':'img-wrap'})
		for img_wrap in img_wraps:
			try:
				bn = os.path.basename(img_wrap.findAll('img')[0]['src'].split('//')[1])
				save_path = os.path.join(save_dir, bn)
				urllib.urlretrieve('https://image.shutterstock.com/z/'+bn, save_path)
			except Exception as e:
				print "exception while getting the image..."
				print str(e)
	except Exception as e:
		print "exception!"
		print str(e)

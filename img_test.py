#import numpy as np
#import skimage.io as io
#import matplotlib.pyplot as plt
#from pycocotools.coco import COCO
#import pylab
from imagenet import VGGLoader

vgg = VGGLoader()
#vgg.process_image('http://mscoco.org/images/%d'%(img['id']))
vgg.process_image('http://mscoco.org/images/376843')
#vgg.process_image('http://farm1.static.flickr.com/8/12567442_838940c1f1.jpg')
#plt.show()

import os
import random
import dircache

dir = '/home/icarus/VAE-im2txt/train2014'
for x in range(0, 100):
	filename = random.choice(dircache.listdir(dir))
	path = os.path.join(dir, filename)
	print filename
	vgg.process_image(path)

import os, urllib
from pycocotools.coco import COCO
import numpy as np

MSCOCO_URL = 'http://mscoco.org/images/'

data_dir = './coco'
data_type = 'train2014'
# Load instances to get categories information
ann_file = '%s/annotations/instances_%s.json' % (data_dir, data_type)

# Define 4 labels we'll work with
labels = ['umbrella']

# Number of images to download per label
NB_IMG = 10

def init_COCO():
    # initialize COCO api for captioning
    return COCO(ann_file)

def download_images(coco):
    for label in labels:
        cat_ids = coco.getCatIds(catNms=label)
        img_ids = coco.getImgIds(catIds=cat_ids)
        imgs = coco.loadImgs(img_ids[:NB_IMG])

        directory = data_dir + '/images/' + label + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        for img in imgs:
            filename = "%s%d.jpeg" % (directory, img['id'])
            if os.path.exists(filename):
                continue
            # Download the image and place it in the file
            urllib.urlretrieve(MSCOCO_URL + '/%d' % img['id'], filename)

if __name__ == "__main__":
    coco = init_COCO()
    download_images(coco)
import os, urllib
from collections import defaultdict

from pycocotools.coco import COCO

MSCOCO_URL = 'http://mscoco.org/images/'

data_dir = './coco'
data_type = 'train2014'
# Load instances to get categories information
ann_file = '%s/annotations/instances_%s.json' % (data_dir, data_type)

# Define 4 labels we'll work with
labels = {'elephant': 0, 'person': 1, 'skis': 2, 'toilet': 3}

# Number of images to download per label
NB_IMG = 100

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

# TODO Finish
def get_images_filenames(coco):
    images_labels = defaultdict([0] * len(labels))

    for label, idx in labels.iteritems():
        cat_ids = coco.getCatIds(catNms=label)
        img_ids = coco.getImgIds(catIds=cat_ids)
        imgs = coco.loadImgs(img_ids[:NB_IMG])

        for img in imgs:
            filename = img['file_name']

            # TODO Finsih
            full_path = ''
            if os.path.exists(full_path):
                images_labels[full_path].append(idx)
            # Download the image and place it in the file

            urllib.urlretrieve(MSCOCO_URL + '/%d' % img['id'], filename)

def get_images_filenames(coco):
    images_labels = defaultdict([0] * len(labels))

    for label, idx in labels.iteritems():
        cat_ids = coco.getCatIds(catNms=label)
        img_ids = coco.getImgIds(catIds=cat_ids)
        imgs = coco.loadImgs(img_ids[:NB_IMG])
	

	print cat_ids	

        for img in imgs:
	    print img
            filename = img['file_name']




if __name__ == "__main__":
    coco = init_COCO()
    download_images(coco)

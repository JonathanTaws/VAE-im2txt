import urllib

import numpy as np
import matplotlib.pyplot as plt
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX
import pickle
import io
import skimage.transform


class VGGLoader:
    def __init__(self):
        model = pickle.load(open('vgg19.pkl'))
        print model.keys()
        print model['mean value']
        self.classes = model['synset words']
        self.mean_image = np.array(model['mean value'])
        self.net = self.build_vgg_model()
        lasagne.layers.set_all_param_values(self.net['prob'], model['param values'])
        print 'Loaded VGG!'

    @staticmethod
    def build_vgg_model():
        net = {}
        net['input'] = InputLayer((None, 3, 224, 224))
        net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=False)
        net['pool3'] = PoolLayer(net['conv3_4'], 2)
        net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=False)
        net['pool4'] = PoolLayer(net['conv4_4'], 2)
        net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
        net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=False)
        net['pool5'] = PoolLayer(net['conv5_4'], 2)
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
        net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
        net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
        net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=1000, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc8'], softmax)

        # Remove the trainable argument from the layers that can potentially have it
        for key, val in net.iteritems():
            if not ('dropout' or 'pool' in key):
                net[key].params[net[key].W].remove("trainable")
                net[key].params[net[key].b].remove("trainable")

        return net


    def prep_image(self, url):
        # ext = url.split('.')[-1]
        im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), 'jpg')
        # Resize so smallest dim = 256, preserving aspect ratio
        h, w, _ = im.shape
        if h < w:
            im = skimage.transform.resize(im, (256, w * 256 / h), preserve_range=True)
        else:
            im = skimage.transform.resize(im, (h * 256 / w, 256), preserve_range=True)
        # Central crop to 224x224
        h, w, _ = im.shape

        im = im[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]


        rawim =np.copy(im).astype('uint8')

        # Shuffle axes to c01
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

        # Convert to BGR
        #im = im[::-1]
        im = im[::-1, :, :]

        # print im

        #   im = im - self.mean_image
        im[0] -= self.mean_image[0]
        im[1] -= self.mean_image[1]
        im[2] -= self.mean_image[2]
        return rawim, floatX(im[np.newaxis])

    def process_image(self, url):
        try:
            rawim, im = self.prep_image(url)

            prob = np.array(lasagne.layers.get_output(self.net['prob'], im, deterministic=True).eval())
            print "calcualted!"
            top5 = np.argsort(prob[0])[-1:-6:-1]

            plt.figure()
            plt.imshow(rawim.astype('uint8'))
            plt.axis('off')
            for n, label in enumerate(top5):
                plt.text(250, 70 + n * 20, '{}. {}'.format(n + 1, self.classes[label]), fontsize=14)
            plt.show()
        except IOError:
            print('bad url: ' + url)

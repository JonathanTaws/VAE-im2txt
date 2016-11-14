import pickle

import lasagne
import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T
from lasagne.layers import DenseLayer, DropoutLayer, NonlinearityLayer, InputLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax, leaky_rectify

from samplelayer import SimpleSampleLayer
from vae import VAEHelper

NUM_LATENT_Z = 2
NUM_CLASSES = 4

TRAIN_FILE = 'x_train_100Samples.p'
VALIDATION_FILE = 'x_train_100SamplesValidation.p'
TARGET_FILE = 'targets_train_100Samples.p'
TARGET_VALIDATION_FILE = 'targets_train_100SamplesValidation.p'

def create_network():
    net = {}

    # VGG Net
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
    #net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=1000, nonlinearity=None)
    #net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    # VAE
    net['enc_vae'] = DenseLayer(net['fc7_dropout'], num_units=128, nonlinearity=leaky_rectify)
    net['muq_vae'] = DenseLayer(net['enc_vae'], num_units=NUM_LATENT_Z, nonlinearity=None)     #mu(x)
    net['logvarq_vae'] = DenseLayer(net['enc_vae'], num_units=NUM_LATENT_Z, nonlinearity=lambda x: T.clip(x,-10,10)) #logvar(x)
    net['z_vae'] = SimpleSampleLayer(mean=net['muq_vae'], log_var=net['logvarq_vae']) # sample a latent representation z \sim q(z|x) = N(mu(x),logvar(x))
    net['in_z_vae'] = InputLayer(shape=(None, NUM_LATENT_Z))
    #net['dec_vae'] = DenseLayer(net['in_z_vae'], num_units=128, nonlinearity=leaky_rectify)

    # Vanilla network
    net['fc8'] = DenseLayer(net['in_z_vae'], num_units=128)
    net['fc8_dropout'] = DropoutLayer(net['fc8'], p=0.5)
    net['fc9'] = DenseLayer(net['fc8_dropout'], num_units=64, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc9'], softmax)

    return net

def set_vgg_params(net):
    model = pickle.load(open('vgg19.pkl'))

    # Remove the trainable argument from the layers that can potentially have it
    for key, val in net.iteritems():
        if not ('dropout' or 'pool' in key):
            net[key].params[net[key].W].remove("trainable")
            net[key].params[net[key].b].remove("trainable")

    lasagne.layers.set_all_param_values(net['prob'], model['param values'])

def training_loop(x_train, x_valid, targets_valid):
    batch_size = 100
    samples_to_process = 1e4
    val_interval = 5e2

    LL_train, KL_train, logpx_train = [],[],[]
    LL_valid, KL_valid, logpx_valid = [],[],[]
    samples_processed = 0
    plt.figure(figsize=(12, 24))
    valid_samples_processed = []

    try:
        while samples_processed < samples_to_process:
            _LL_train, _KL_train, _logpx_train = [],[],[]
            idxs = np.random.choice(range(x_train.shape[0]), size=(batch_size), replace=False)
            x_batch = x_train[idxs]
            out = f_train(x_batch)
            samples_processed += batch_size

            if samples_processed % val_interval == 0:
                valid_samples_processed += [samples_processed]
                out = f_eval(x_train)
                LL_train += [out[0]]
                logpx_train += [out[1][:,0]] # or could mean the log_px_given_z, KL_qp in ll function
                KL_train += [out[2][:,0]]

                out = f_eval(x_valid)
                LL_valid += [out[0]]
                logpx_valid += [out[1][:,0]] #just pick a single sample, or could mean the log_px_given_z, KL_qp in ll function
                KL_valid += [out[2][:,0]]

                z_eval = f_z(x_valid)[0]
                x_sample = f_sample(np.random.normal(size=(100, NUM_LATENT_Z)).astype('float32'))[0]
                x_recon = f_recon(x_valid)[0]

                plt.subplot(NUM_CLASSES+2,2,1)
                plt.legend(['LL', 'log(p(x))'], loc=2)
                plt.xlabel('Updates')
                plt.plot(valid_samples_processed, LL_train, color="black")
                plt.plot(valid_samples_processed, logpx_train, color="red")
                plt.plot(valid_samples_processed, LL_valid, color="black", linestyle="--")
                plt.plot(valid_samples_processed, logpx_valid, color="red", linestyle="--")
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.grid('on')

                plt.subplot(NUM_CLASSES+2,2,2)
                plt.cla()
                plt.xlabel('z0'), plt.ylabel('z1')
                color = iter(plt.get_cmap('brg')(np.linspace(0, 1.0, NUM_CLASSES)))
                for i in range(NUM_CLASSES):
                    clr = next(color)
                    plt.scatter(z_eval[targets_valid==i, 0], z_eval[targets_valid==i, 1], c=clr, s=5., lw=0, marker='o', )
                plt.grid('on')

                #plt.savefig("out52.png")
                #display(Image(filename="out52.png"))
                #clear_output(wait=True)

                plt.subplot(NUM_CLASSES+2,2,3)
                plt.legend(['KL(q||p)'])
                plt.xlabel('Updates')
                plt.plot(valid_samples_processed, KL_train, color="blue")
                plt.plot(valid_samples_processed, KL_valid, color="blue", linestyle="--")
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.grid('on')

                plt.subplot(NUM_CLASSES+2,2,4)
                plt.cla()
                plt.title('Samples')
                plt.axis('off')
                idx = 0
                canvas = np.zeros((28*10, 10*28))
                for i in range(10):
                    for j in range(10):
                        canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_sample[idx].reshape((28, 28))
                        idx += 1
                plt.imshow(canvas, cmap='gray')

                c=0
                for k in range(5, 5 + NUM_CLASSES*2, 2):
                    plt.subplot(NUM_CLASSES+2,2,k)
                    plt.cla()
                    plt.title('Inputs for %i' % c)
                    plt.axis('off')
                    idx = 0
                    canvas = np.zeros((28*10, 10*28))
                    for i in range(10):
                        for j in range(10):
                            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_valid[targets_valid==c][idx].reshape((28, 28))
                            idx += 1
                    plt.imshow(canvas, cmap='gray')

                    plt.subplot(NUM_CLASSES+2,2,k+1)
                    plt.cla()
                    plt.title('Reconstructions for %i' % c)
                    plt.axis('off')
                    idx = 0
                    canvas = np.zeros((28*10, 10*28))
                    for i in range(10):
                        for j in range(10):
                            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_recon[targets_valid==c][idx].reshape((28, 28))
                            idx += 1
                    plt.imshow(canvas, cmap='gray')
                    c += 1

    except KeyboardInterrupt:
        pass

def load_data():
    x_train = pickle.load(TRAIN_FILE)
    x_valid = pickle.load(VALIDATION_FILE)
    targets_valid = pickle.load(TARGET_VALIDATION_FILE)

    return x_train, x_valid, targets_valid

if __name__ == '__main__':
    # Create network
    net = create_network()

    f_train, f_eval, f_z, f_sample, f_recon = VAEHelper.create_theano_functions(net)

    # TODO : Call function to get the data
    x_train, x_valid, targets_valid = load_data()

    training_loop(x_train, x_valid, targets_valid)








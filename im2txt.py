from lasagne.layers import DenseLayer, DropoutLayer, NonlinearityLayer
from lasagne.nonlinearities import softmax
from imagenet import VGGLoader
from vae import VAEBuilder
import numpy as np
import matplotlib.pyplot as plt

NUM_LATENT_Z = 2
NUM_CLASSES = 4

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

if __name__ == '__main__':
    # Load VGG and create net
    net = VGGLoader.build_vgg_model()

    # Add VAE layers
    vae = VAEBuilder(num_latent_z=NUM_LATENT_Z)
    last_layer_vae = vae.add_layers(net, net['fc7_dropout'])

    # Add vanilla net as decoder
    VAEBuilder.add_vanilla_net(net, last_layer_vae)

    f_train, f_eval, f_z, f_sample, f_recon = VAEBuilder.create_theano_functions(net)








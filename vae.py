import lasagne
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer, NonlinearityLayer, get_output, get_all_params
from lasagne.nonlinearities import leaky_rectify, softmax
import theano.tensor as T
import theano
from samplelayer import SimpleSampleLayer

class VAEBuilder:

    def __init__(self, num_latent_z):
        self.num_latent_z = num_latent_z

    def add_layers(self, net, prev_layer):
        net['enc_vae'] = DenseLayer(prev_layer, num_units=128, nonlinearity=leaky_rectify)
        net['muq_vae'] = DenseLayer(net['enc_vae'], num_units=self.num_latent_z, nonlinearity=None)     #mu(x)
        net['logvarq_vae'] = DenseLayer(net['enc_vae'], num_units=self.num_latent_z, nonlinearity=lambda x: T.clip(x,-10,10)) #logvar(x)
        net['z_vae'] = SimpleSampleLayer(mean=net['muq_vae'], log_var=net['logvarq_vae']) # sample a latent representation z \sim q(z|x) = N(mu(x),logvar(x))
        net['in_z_vae'] = InputLayer(shape=(None, self.num_latent_z))
        #net['dec_vae'] = DenseLayer(net['in_z_vae'], num_units=128, nonlinearity=leaky_rectify)

        last_layer = net['in_z_vae']

        return last_layer

    @staticmethod
    def add_vanilla_net(net, prev_layer):
        net['fc7'] = DenseLayer(prev_layer, num_units=128)
        net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
        net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=64, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    @staticmethod
    def create_theano_functions(net):
        sym_x = T.matrix('x')
        sym_z = T.matrix('z')

        z_train, muq_train, logvarq_train = get_output([net['z_vae'], net['muq_vae'], net['logvarq_vae']],
                                                       {net['input']: sym_x}, deterministic=False)
        prob_train = get_output(net['prob'], {net['in_z_vae']: z_train}, deterministic=False)

        z_eval, muq_eval, logvarq_eval = get_output([net['z_vae'], net['muq_vae'], net['logvarq_vae']],
                                                    {net['input']: sym_x}, deterministic=True)
        prob_eval = get_output(net['prob'], {net['in_z_vae']: z_eval}, deterministic=True)

        prob_sample = get_output(net['prob'], {net['in_z_vae']: sym_z}, deterministic=True)

        LL_train, logpx_train, KL_train = VAEBuilder.LogLikelihood(prob_train, sym_x, muq_train, logvarq_train)
        LL_eval, logpx_eval, KL_eval = VAEBuilder.LogLikelihood(prob_eval, sym_x, muq_eval, logvarq_eval)

        all_params = get_all_params([net['z_vae'],net['prob']],trainable=True)

        # Let Theano do its magic and get all the gradients we need for training
        all_grads = T.grad(-LL_train, all_params)

        # Set the update function for parameters. The Adam optimizer works really well with VAEs.
        updates = lasagne.updates.adam(all_grads, all_params, learning_rate=1e-2)

        f_train = theano.function(inputs=[sym_x],
                                  outputs=[LL_train, logpx_train, KL_train],
                                  updates=updates)

        f_eval = theano.function(inputs=[sym_x],
                                 outputs=[LL_eval, logpx_eval, KL_eval])

        f_z = theano.function(inputs=[sym_x],
                              outputs=[z_eval])

        f_sample = theano.function(inputs=[sym_z],
                                   outputs=[prob_sample])

        f_recon = theano.function(inputs=[sym_x],
                                  outputs=[prob_eval])

        return f_train, f_eval, f_z, f_sample, f_recon

    @staticmethod
    def LogLikelihood(mux,x,muq,logvarq):
        log_px_given_z = VAEBuilder.log_bernoulli(x, mux, eps=1e-6).sum(axis=1).mean() #note that we sum the latent dimension and mean over the samples
        KL_qp = VAEBuilder.kl_normal2_stdnormal(muq, logvarq).sum(axis=1).mean() # * 0 # To ignore the KL term
        LL = log_px_given_z - KL_qp
        return LL, log_px_given_z, KL_qp

    @staticmethod
    def log_bernoulli(x, p, eps=0.0):
        p = T.clip(p, eps, 1.0 - eps)
        return -T.nnet.binary_crossentropy(p, x)

    @staticmethod
    def kl_normal2_stdnormal(mean, log_var):
        return -0.5*(1 + log_var - mean**2 - T.exp(log_var))
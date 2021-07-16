"""
Author: Alexandra Lee
Date Created: 30 August 2019

Helper functions for running variational autoencoder `vae.py`.
"""

from keras.callbacks import Callback
from keras import metrics
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf
import numpy as np

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see:
# https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# Functions
#
# Based on publication by Way et. al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5728678/)
# Github repo (https://github.com/greenelab/tybalt/blob/master/scripts/vae_pancancer.py)


# Function for reparameterization trick to make model differentiable


def sampling_maker(epsilon_std):
    def sampling(args):
        # Function with args required for Keras Lambda function
        z_mean, z_log_var = args

        # Draw epsilon of the same shape from a standard normal distribution
        epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.0, stddev=epsilon_std)

        # The latent vector is non-deterministic and differentiable
        # in respect to z_mean and z_log_var
        z = z_mean + K.exp(z_log_var / 2) * epsilon
        return z

    return sampling


class CustomVariationalLayer(Layer):
    """
    Define a custom layer that learns and performs the training
    """

    def __init__(self, original_dim, z_log_var_encoded, z_mean_encoded, beta, **kwargs):
        # https://keras.io/layers/writing-your-own-keras-layers/
        self.is_placeholder = True
        self.original_dim = original_dim
        self.z_log_var_encoded = z_log_var_encoded
        self.z_mean_encoded = z_mean_encoded
        self.beta = beta

        super(CustomVariationalLayer, self).__init__(**kwargs)

    def kl_loss(self):
        return -0.5 * K.sum(
            1
            + self.z_log_var_encoded
            - K.square(self.z_mean_encoded)
            - K.exp(self.z_log_var_encoded),
            axis=-1,
        )

    def reconstruction_loss(self, x_input, x_decoded):
        return self.original_dim * metrics.binary_crossentropy(x_input, x_decoded)

    # From https://github.com/theislab/dca/blob/8210adf66acb7a55da6fcbb1915d40a188a5420f/dca/loss.py
    def _nan2inf(self, x):
        return tf.where(tf.math.is_nan(x), tf.zeros_like(x) + np.inf, x)

    # This code is based on publication: https://www.nature.com/articles/s41467-018-07931-2
    # https://github.com/theislab/dca/blob/master/dca/loss.py
    # From https://github.com/theislab/dca/blob/8210adf66acb7a55da6fcbb1915d40a188a5420f/dca/loss.py#L116

    # The zero-inflated negative binomial (ZINB) distribution models highly sparse
    # and overdispersed count data

    # ZINB is a mixture model that is composed of
    # 1. A point mass at 0 to represent the excess of 0's
    # 2. A NB distribution to represent the count distribution

    # Params of ZINB conditioned on the input data are estimated
    # Params include the mean and dispersion parameters of the NB component
    # (μ and θ) and the mixture coefficient that represents the weight of
    # the point mass (π)
    # These params should be updated by the network

    def zinb_loss(self, y_true, y_pred):
        # pi is the probability that the count is 0
        pi = 0.9
        # ridge_lambda is the strength of the L1/L2 regularization
        # what does this mean???
        ridge_lambda = 0.0
        # scale_factor scales the nbinom mean before the
        # calculation of the loss to balance the
        # learning rates of theta and network weights
        # So what does high value mean???
        scale_factor = 1.0
        # eps is???
        # Scale for eps???
        eps = 1e-10
        # theta is the dispersion of the NB distribution
        # minimum of theta is 1e6
        theta = 1e10

        # Commenting these out and hard coding variables in for now
        # scale_factor = self.scale_factor
        # eps = self.eps

        # reuse existing NB neg.log.lik.
        # mean is always False here, because everything is calculated
        # element-wise. we take the mean only in the end

        # Replace `super().loss(y_true, y_pred)` with code directly from loss()
        # https://github.com/theislab/dca/blob/8210adf66acb7a55da6fcbb1915d40a188a5420f/dca/loss.py#L72
        # Getting an error that 'super hase no attribute loss'
        # nb_case = super().loss(y_true, y_pred, mean=False) - tf.math.log(1.0 - pi + eps)

        t1 = tf.math.lgamma(theta + eps) + tf.math.lgamma(y_true + 1.0) - tf.math.lgamma(y_true + theta + eps)
        t2 = (theta + y_true) * tf.math.log(1.0 + (y_pred / (theta + eps))) + (y_true * (tf.math.log(theta + eps) - tf.math.log(y_pred + eps)))
        nb_case = t1 + t2 - tf.math.log(1.0 - pi + eps)

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32) * scale_factor
        # theta = tf.minimum(self.theta, 1e6)

        zero_nb = tf.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -tf.math.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
        ridge = ridge_lambda * tf.square(pi)
        result += ridge

        result = tf.reduce_mean(result)

        result = self._nan2inf(result)

        # if self.debug:
        #    tf.summary.histogram('nb_case', nb_case)
        #    tf.summary.histogram('zero_nb', zero_nb)
        #    tf.summary.histogram('zero_case', zero_case)
        #    tf.summary.histogram('ridge', ridge)

        return result

    def vae_loss(self, y_true, y_pred):
        # reconstruction_loss = self.reconstruction_loss(x_input, x_decoded)
        kl_loss = self.kl_loss()
        zinb_loss = self.zinb_loss(y_true, y_pred)

        return K.mean(zinb_loss + (K.get_value(self.beta) * kl_loss))

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        self.add_loss(loss, inputs=inputs)

        kl_loss = self.kl_loss()
        self.add_metric(kl_loss, name="kl_loss")
        zinb_loss = self.zinb_loss(x, x_decoded)
        self.add_metric(zinb_loss, name="zinb_loss")
        # We won't actually use the output.
        return x


class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa

    # Behavior on each epoch
    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)

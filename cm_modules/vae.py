"""
Author: Alexandra Lee
Updated October 2018

Scripts to train 2-layer variational autoencoder.
"""
import pandas as pd
import tensorflow as tf

# To ensure reproducibility using Keras during development
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import numpy as np
import os

from keras import backend as K
from keras.layers import Input, Dense, Lambda, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras import optimizers
from cm_modules.helper_vae import sampling_maker, CustomVariationalLayer, WarmUpCallback


# This function is scrapped from Georgia Doing's repository:
# https://github.com/georgiadoing/seqADAGE/blob/master/Py/run_count_autoencoder.py
# This code is based on publication: https://www.nature.com/articles/s41467-018-07931-2
# https://github.com/theislab/dca/blob/master/dca/loss.py
# def _nan2inf(x):
#    return tf.where(tf.math.is_nan(x), tf.zeros_like(x) + np.inf, x)
# def zinbl2(y_true, y_pred):
#    y_true = tf.cast(y_true, tf.float32)
#    y_pred = tf.cast(y_pred, tf.float32)
#    eps = 1e-10
#    theta = 1e6
# pi = y_pred / y_pred + math.exp
# pi = 0.5
# ridge_lambda = 0
# t1 = tf.math.lgamma(1e6 + 1e-10) + tf.math.lgamma(y_true + 1.0) - tf.math.lgamma(y_true + 1e6 + 1e-10)
# t2 = (1e6 + y_true) * tf.math.log(1.0 + (y_pred / (1e6 + 1e-10))) + (y_true * (tf.math.log(1e6 + 1e-10) - tf.math.log(y_pred + 1e-10)))

# nb_case = t1 + t2 - tf.math.log(1.0 - pi + eps)
# zero_nb = tf.pow(theta / (theta + y_pred + eps), theta)
# zero_case = -tf.math.log(pi + ((1.0 - pi) * zero_nb) + eps)
# result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
# ridge = ridge_lambda * tf.square(pi)
# result += ridge

# final = tf.reduce_mean(result)
# return _nan2inf(final)


def run_tybalt_training(
    expression_data,
    learning_rate,
    batch_size,
    epochs,
    kappa,
    intermediate_dim,
    latent_dim,
    epsilon_std,
    validation_frac,
):
    """
    Create and train a VAE based on the Tybalt paper.
    This function does the heavy lifting for `tybalt_2layer_model`, while the calling function
    handles file IO

    Arguments
    ---------
    expression_data: pandas.dataframe
        The expression data to be used to train the VAE

    learning_rate: float
        Step size used for gradient descent. In other words, it's how quickly
        the  methods is learning

    batch_size: int
        Training is performed in batches. So this determines the number of
        samples to consider at a given time.

    epochs: int
        The number of times to train over the entire input dataset.

    kappa: float
        How fast to linearly ramp up KL loss

    intermediate_dim: int
        Size of the hidden layer

    latent_dim: int
        Size of the bottleneck layer

    epsilon_std: float
        Standard deviation of Normal distribution to sample latent space

    validation_frac: float
        Percentage of total dataset to set aside to use as a validation set.

    Returns
    -------
    encoder: keras.models.Model
        The encoder half of the VAE. `encoder` takes in a (samples x genes) dataframe of
        gene expression data and encodes it into a latent space

    decoder: keras.models.Model
        The decoder half of the VAE. `decoder` takes a dataframe of means and standard deviations
        and uses them to simulate gene expression data close to the distribution of a
        a set of experiments from normalized_data

    hist: keras.callbacks.History
        The history object containing training information returned when fitting the VAE
    """
    original_dim = expression_data.shape[1]
    beta = K.variable(0)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details,
    # see:
    # https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # Data initalizations

    # Split 10% test set randomly
    test_set_percent = validation_frac
    expression_test_df = expression_data.sample(frac=test_set_percent)
    expression_train_df = expression_data.drop(expression_test_df.index)

    # Create a placeholder for an encoded (original-dimensional)
    expression_input = Input(shape=(original_dim,))

    # Architecture of VAE

    # ENCODER

    # Input layer is compressed into a mean and log variance vector of size
    # `latent_dim`. Each layer is initialized with glorot uniform weights and
    # each step (dense connections, batch norm,and relu activation) are
    # funneled separately
    # Each vector of length `latent_dim` are connected to the expression input
    # tensor

    # "z_mean_dense_linear" is the encoded representation of the input
    # Take as input arrays of shape (*, original dim) and output arrays of
    # shape (*, latent dim)
    # Combine input from previous layer using linear sum
    # Normalize the activations (combined weighted nodes of the previous layer)
    # Transformation that maintains the mean activation close to 0 and the
    # activation standard deviation close to 1.
    # Apply ReLU activation function to combine weighted nodes from previous
    # layer
    #   relu = threshold cutoff (cutoff value will be learned)
    #   ReLU function filters noise

    # X is encoded using Q(z|X) to yield mu(X), sigma(X) that describes latent
    # space distribution
    hidden_dense_linear = Dense(intermediate_dim, kernel_initializer="glorot_uniform")(
        expression_input
    )
    hidden_dense_batchnorm = BatchNormalization()(hidden_dense_linear)
    hidden_encoded = Activation("relu")(hidden_dense_batchnorm)

    # Note:
    # Normalize and relu filter at each layer adds non-linear component
    # (relu is non-linear function)
    # If architecture is layer-layer-normalization-relu then the computation
    # is still linear
    # Add additional layers in triplicate
    z_mean_dense_linear = Dense(latent_dim, kernel_initializer="glorot_uniform")(
        hidden_encoded
    )
    z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
    z_mean_encoded = Activation("relu")(z_mean_dense_batchnorm)

    z_log_var_dense_linear = Dense(latent_dim, kernel_initializer="glorot_uniform")(
        expression_input
    )
    z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
    z_log_var_encoded = Activation("relu")(z_log_var_dense_batchnorm)

    # Customized layer
    # Returns the encoded and randomly sampled z vector
    # Takes two keras layers as input to the custom sampling function layer with a
    # latent_dim` output
    #
    # sampling():
    # randomly sample similar points z from the latent normal distribution that is assumed to generate the data,
    # via z = z_mean + exp(z_log_sigma) * epsilon, where epsilon is a random normal tensor
    # z ~ Q(z|X)
    # Note: there is a trick to reparameterize to standard normal distribution so that the space is differentiable and
    # therefore gradient descent can be used
    #
    # Returns the encoded and randomly sampled z vector
    # Takes two keras layers as input to the custom sampling function layer with a
    # latent_dim` output
    z = Lambda(sampling_maker(epsilon_std), output_shape=(latent_dim,))(
        [z_mean_encoded, z_log_var_encoded]
    )

    # DECODER

    # The decoding layer is much simpler with a single layer glorot uniform
    # initialized and sigmoid activation
    # Reconstruct P(X|z)
    decoder = Sequential()
    decoder.add(Dense(intermediate_dim, activation="relu", input_dim=latent_dim))
    decoder.add(Dense(original_dim, activation="sigmoid"))
    expression_reconstruct = decoder(z)

    # CONNECTIONS
    # fully-connected network
    adam = optimizers.Adam(lr=learning_rate)
    vae_layer = CustomVariationalLayer(
        original_dim, z_log_var_encoded, z_mean_encoded, beta
    )([0.5, expression_input, expression_reconstruct, 0.0])
    vae = Model(expression_input, vae_layer)

    # UPDATE LOSS HERE
    vae.compile(optimizer=adam, loss=None, loss_weights=[beta])
    # vae.compile(optimizer=adam, loss=zinbl2)

    # Training

    # fit Model
    # hist: record of the training loss at each epoch
    hist = vae.fit(
        np.array(expression_train_df),
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(np.array(expression_test_df), None),
        callbacks=[WarmUpCallback(beta, kappa)],
    )

    # Use trained model to make predictions
    encoder = Model(expression_input, z_mean_encoded)

    return encoder, decoder, hist


def tybalt_2layer_model(
    learning_rate,
    batch_size,
    epochs,
    kappa,
    intermediate_dim,
    latent_dim,
    epsilon_std,
    rnaseq,
    base_dir,
    dataset_name,
    NN_name,
    validation_frac,
):
    """
    Train 2-layer Tybalt model using input dataset

    Arguments
    ----------
    learning_rate: float
        Step size used for gradient descent. In other words, it's how quickly
        the  methods is learning

    batch_size: int
        Training is performed in batches. So this determines the number of
        samples to consider at a given time.

    epochs: int
        The number of times to train over the entire input dataset.

    kappa: float
        How fast to linearly ramp up KL loss

    intermediate_dim: int
        Size of the hidden layer

    latent_dim: int
        Size of the bottleneck layer

    epsilon_std: float
        Standard deviation of Normal distribution to sample latent space

    rnaseq: pandas.dataframe
        Gene expression data

    base_dir: str
        Root directory containing analysis subdirectories

    dataset_name: str
        Name of analysis directory

    NN_name: str
        Neural network architecture of VAE.
        Format NN_<intermediate_dim>_<latent_dim>

    validation_frac: float
        Percentage of total dataset to set aside to use as a validation set.

    Returns
    --------
    model_decoder_filnamee, weights_decoder_filename: .h5 file
        Files used to generate decoding neural networks to use in downstream
        analysis

    model_encoder_filename, weights_encoder_filename: .h5 file
        Files used to generate encoding neural networks to use in downstream
        analysis

    """
    # Initialize hyper parameters

    stat_filename = os.path.join(
        base_dir,
        dataset_name,
        "logs",
        NN_name,
        "tybalt_2layer_{}latent_stats.tsv".format(latent_dim),
    )

    hist_plot_filename = os.path.join(
        base_dir,
        dataset_name,
        "logs",
        NN_name,
        "tybalt_2layer_{}latent_hist.svg".format(latent_dim),
    )

    model_encoder_filename = os.path.join(
        base_dir,
        dataset_name,
        "models",
        NN_name,
        "tybalt_2layer_{}latent_encoder_model.h5".format(latent_dim),
    )

    weights_encoder_filename = os.path.join(
        base_dir,
        dataset_name,
        "models",
        NN_name,
        "tybalt_2layer_{}latent_encoder_weights.h5".format(latent_dim),
    )

    model_decoder_filename = os.path.join(
        base_dir,
        dataset_name,
        "models",
        NN_name,
        "tybalt_2layer_{}latent_decoder_model.h5".format(latent_dim),
    )

    weights_decoder_filename = os.path.join(
        base_dir,
        dataset_name,
        "models",
        NN_name,
        "tybalt_2layer_{}latent_decoder_weights.h5".format(latent_dim),
    )

    encoder, decoder_model, hist = run_tybalt_training(
        rnaseq,
        learning_rate,
        batch_size,
        epochs,
        kappa,
        intermediate_dim,
        latent_dim,
        epsilon_std,
        validation_frac,
    )

    encoded_rnaseq_df = encoder.predict_on_batch(rnaseq)
    encoded_rnaseq_df = pd.DataFrame(encoded_rnaseq_df, index=rnaseq.index)

    encoded_rnaseq_df.columns.name = "sample_id"
    encoded_rnaseq_df.columns = encoded_rnaseq_df.columns + 1

    # Visualize training performance
    history_df = pd.DataFrame(hist.history)
    ax = history_df.plot(y="loss", label="Training loss")
    history_df.plot(y="val_loss", label="Validation loss", ax=ax)
    ax.set_xlabel("Epochs", fontsize="xx-large", family="sans-serif")
    ax.set_ylabel("Loss", fontsize="xx-large", family="sans-serif")
    fig = ax.get_figure()
    fig.savefig(hist_plot_filename, dpi=300)

    del ax, fig

    # Output

    # Save training performance
    history_df = pd.DataFrame(hist.history)
    history_df = history_df.assign(learning_rate=learning_rate)
    history_df = history_df.assign(batch_size=batch_size)
    history_df = history_df.assign(epochs=epochs)
    history_df = history_df.assign(kappa=kappa)
    history_df.to_csv(stat_filename, sep="\t", index=False)

    # Save models
    # (source) https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    # Save encoder model
    encoder.save(model_encoder_filename)

    # serialize weights to HDF5
    encoder.save_weights(weights_encoder_filename)

    # Save decoder model
    # (source) https://github.com/greenelab/tybalt/blob/master/scripts/nbconverted/tybalt_vae.py
    # can generate from any sampled z vector
    decoder_input = Input(shape=(latent_dim,))
    _x_decoded_mean = decoder_model(decoder_input)
    decoder = Model(decoder_input, _x_decoded_mean)

    decoder.save(model_decoder_filename)

    # serialize weights to HDF5
    decoder.save_weights(weights_decoder_filename)

    # Save weight matrix:  how each gene contribute to each feature
    # build a generator that can sample from the learned distribution
    # can generate from any sampled z vector
    decoder_input = Input(shape=(latent_dim,))
    x_decoded_mean = decoder_model(decoder_input)
    decoder = Model(decoder_input, x_decoded_mean)
    weights = []
    for layer in decoder.layers:
        weights.append(layer.get_weights())

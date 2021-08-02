"""
Author: Alexandra Lee
Updated October 2018

Scripts to train 2-layer variational autoencoder.
"""
import tensorflow as tf

# To ensure reproducibility using Keras during development
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import numpy as np
import os
import random

from keras import backend as K
from .network import Autoencoder, ZINBAutoencoder
from .train import train
from .network import AE_types
from . import io
import scanpy as sc


# Update VAE model to use count based loss to account for the sparsity in the data
# This new model is based on Deep Count Autoencoder: https://www.nature.com/articles/s41467-018-07931-2
# More details: https://docs.google.com/presentation/d/1Q_0BUbfg51OicxY4MdI0IwhdhFfJmzX0f8VyuyGNXrw/edit#slide=id.ge45eb3c133_0_56
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

    # set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '0'

    # Format input data
    print(isinstance(expression_data, sc.AnnData))
    print(isinstance(expression_data, str))
    adata = io.read_dataset(
        expression_data,
        transpose=True,  # assume gene x cell by default
        check_counts=True,
        test_split=True
        )

    original_dim = expression_data.shape[1]

    # Define model architecture
    net = AE_types['zinb-conddisp'](
        input_size=original_dim,
        output_size=original_dim,
        hidden_size=(intermediate_dim, latent_dim, intermediate_dim),
        l2_coef=0.,
        l1_coef=0.,
        l2_enc_coef=0.,
        l1_enc_coef=0.,
        ridge=0.,
        hidden_dropout=0.,
        input_dropout=0.,
        batchnorm=True,
        activation='relu',
        init='glorot_uniform',
        file_path=None,
        debug=True
        )
    net.save()
    net.build()

    # Train model using ZINB loss
    # Expect input that is gene x cell
    """losses = train(
        expression_data.T,
        net,
        output_dir=None,
        learning_rate=learning_rate,
        epochs=epochs,
        early_stop=15,
        reduce_lr=10,
        output_subset=None,
        optimizer='Adam',
        clip_grad=5.,
        save_weights=True,
        tensorboard=False
        )"""


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

    model = run_tybalt_training(
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

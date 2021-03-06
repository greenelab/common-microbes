"""
Author: Alexandra Lee
Date Created: 11 March 2020

Scripts related to training the VAE including
1. Normalizing gene expression data
2. Wrapper function to input training parameters and run vae
training in `vae.tybalt_2layer_model`
"""

from cm_modules import utils
import os
import pickle
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import random
import warnings
from .network import AE_types
from . import io
from .train import train
# import scanpy as sc


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


# with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    fxn()


def set_all_seeds(seed_val=42):
    """
    This function sets all seeds to get reproducible VAE trained
    models.
    """

    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

    os.environ["PYTHONHASHSEED"] = "0"

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(seed_val)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    random.seed(seed_val)
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    tf.set_random_seed(seed_val)


def normalize_expression_data(base_dir, config_filename, raw_input_data_filename, normalized_data_filename):
    """
    0-1 normalize the expression data.

    Arguments
    ----------
    base_dir: str
        Root directory containing analysis subdirectories

    config_filename: str
        File containing user defined parameters

    raw_input_data_filename: str
        File containing raw expression data

    normalize_data_filename:
        Output file containing normalized expression data
    """
    # Read in config variables
    params = utils.read_config(config_filename)

    # Read data
    data = pd.read_csv(raw_input_data_filename, header=0, sep="\t", index_col=0)
    print(
        "input: dataset contains {} samples and {} genes".format(
            data.shape[0], data.shape[1]
        )
    )

    # 0-1 normalize per gene
    scaler = preprocessing.MinMaxScaler()
    data_scaled_df = scaler.fit_transform(data)
    data_scaled_df = pd.DataFrame(
        data_scaled_df, columns=data.columns, index=data.index
    )

    print(
        "Output: normalized dataset contains {} samples and {} genes".format(
            data_scaled_df.shape[0], data_scaled_df.shape[1]
        )
    )

    # Save scaler transform
    scaler_filename = params["scaler_transform_filename"]

    outfile = open(scaler_filename, "wb")
    pickle.dump(scaler, outfile)
    outfile.close()

    # Save scaled data
    data_scaled_df.to_csv(normalized_data_filename, sep="\t", compression="xz")


def train_vae(config_filename, input_data_filename):
    """
    Trains VAE model using parameters set in config file

    Arguments
    ----------
    config_filename: str
        File containing user defined parameters

    input_data_filename: str
        File path corresponding to input dataset to use
    """

    # Read in config variables
    params = utils.read_config(config_filename)

    # Load parameters
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
    dataset_name = params["dataset_name"]
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    kappa = params["kappa"]
    intermediate_dim = params["intermediate_dim"]
    latent_dim = params["latent_dim"]
    epsilon_std = params["epsilon_std"]
    train_architecture = params["NN_architecture"]
    validation_frac = params["validation_frac"]

    # set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '0'

    # Format input data
    adata = io.read_dataset(
        input_data_filename,
        transpose=False,  # assume gene x cell by default
        check_counts=False,
        test_split=True
    )
    print("Successfully read in data")
    print(adata.X.shape)

    # Normalize input data
    # Samples below min count were removed
    adata = io.normalize(
        adata,
        size_factors=False,
        normalize_input=True)

    print("Normalized input data")
    print(adata.X.shape)

    # Want cell x gene input
    # Our dataset is sample x microbe so no need to transform
    # input_data = pd.read_csv(input_data_filename, header=0, sep="\t", index_col=0)
    original_dim = adata.X.shape[1]
    print(
        "input dataset contains {} rows and {} columns".format(
            adata.X.shape[0], adata.X.shape[1]
        )
    )

    # Define model architecture
    net = AE_types['zinb-conddisp'](
        input_size=original_dim,
        output_size=original_dim,
        hidden_size=[intermediate_dim, latent_dim, intermediate_dim],
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
    print("built network")

    # Train model using ZINB loss
    # Expect input that is gene x cell
    losses = train(
        adata[adata.obs.dca_split == 'train'],
        net,
        output_dir=None,
        learning_rate=learning_rate,
        epochs=epochs,
        early_stop=15,
        reduce_lr=10,
        output_subset=None,
        optimizer_val='Adam',
        clip_grad=5.,
        save_weights=True,
        tensorboard=False
        )

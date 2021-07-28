# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1+dev
#   kernelspec:
#     display_name: Python [conda env:common_microbe] *
#     language: python
#     name: conda-env-common_microbe-py
# ---

# # Train VAE model
#
# This notebook will first try to train the current VAE model before modifying the loss function to work with count data. Specifically, the loss function uses the loss from defined in [Eraslan et al.](https://www.nature.com/articles/s41467-018-07931-2). This publication uses the zero-inflated negative binomial (ZINB) distribution, which models highly sparse and overdispersed count data. ZINB is a mixture model that is composed of
#    1. A point mass at 0 to represent the excess of 0's
#    2. A NB distribution to represent the count distribution
#
# Params of ZINB conditioned on the input data are estimated. These params include the mean and dispersion parameters of the NB component (μ and θ) and the mixture coefficient that represents the weight of the point mass (π)
#
# We adopted code from: https://github.com/theislab/dca/blob/master/dca/loss.py
#
# More details about the new model can be found: https://docs.google.com/presentation/d/1Q_0BUbfg51OicxY4MdI0IwhdhFfJmzX0f8VyuyGNXrw/edit#slide=id.ge45eb3c133_0_56

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
import os
import matplotlib.pyplot as plt
import pandas as pd
from cm_modules import paths, utils, train_vae_modules
import scanpy as sc
import anndata

# +
# Set seeds to get reproducible VAE trained models
# train_vae_modules.set_all_seeds()

# +
base_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))

# Read in config variables
config_filename = os.path.abspath(
    os.path.join(base_dir, "test_vae_training", "config_current_vae.tsv")
)

params = utils.read_config(config_filename)

dataset_name = params["dataset_name"]

raw_compendium_filename = params["raw_compendium_filename"]
normalized_compendium_filename = params["normalized_compendium_filename"]
# -

raw_compendium = pd.read_csv(raw_compendium_filename, sep="\t", index_col=0, header=0)

print(raw_compendium.shape)
raw_compendium.head()

raw_compendium.T.to_csv(
    os.path.join(paths.LOCAL_DATA_DIR, "raw_microbiome_transposed.tsv"), sep="\t"
)

# +
# TO DO:
# In the DCA paper, they log2 transformed and z-score normalized their data

# Try normalzing the data
# Here we are normalizing the microbiome count data per taxon
# so that each taxon is in the range 0-1
train_vae_modules.normalize_expression_data(
    base_dir, config_filename, raw_compendium_filename, normalized_compendium_filename
)
# -

# test1 = pd.read_csv(raw_compendium_filename, sep="\t")
test2 = pd.read_csv(normalized_compendium_filename, sep="\t", index_col=0, header=0)

# +
# test1.head()
# -

test2.shape

test2.head()

# Convert input to anndata object
test2_anndata = anndata.AnnData(test2)

# Save
raw_compendium_anndata_filename = os.path.join(
    paths.LOCAL_DATA_DIR, "raw_microbiome_transposed_anndata.h5ad"
)
test2_anndata.write(raw_compendium_anndata_filename)

# +
# Create VAE directories if needed
output_dirs = [
    os.path.join(base_dir, dataset_name, "models"),
    os.path.join(base_dir, dataset_name, "logs"),
]

NN_architecture = params["NN_architecture"]

# Check if NN architecture directory exist otherwise create
for each_dir in output_dirs:
    sub_dir = os.path.join(each_dir, NN_architecture)
    os.makedirs(sub_dir, exist_ok=True)
# -

# Train VAE on new compendium data
train_vae_modules.train_vae(config_filename, raw_compendium_anndata_filename)

# +
# Plot training and validation loss separately
# stat_logs_filename = "logs/DCA/tybalt_2layer_30latent_stats.tsv"

# stats = pd.read_csv(stat_logs_filename, sep="\t", index_col=None, header=0)

# +
# plt.plot(stats["loss"])

# +
# plt.plot(stats["val_loss"], color="orange")

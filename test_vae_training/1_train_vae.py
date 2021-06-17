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
#     display_name: Python [conda env:microbe] *
#     language: python
#     name: conda-env-microbe-py
# ---

# # Train VAE model
#
# This notebook will first try to train the current VAE model before modifying the loss function to work with count data

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
import os
import matplotlib.pyplot as plt
import pandas as pd
from ponyo import utils, train_vae_modules

# Set seeds to get reproducible VAE trained models
train_vae_modules.set_all_seeds()

# +
base_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))

# Read in config variables
config_filename = os.path.abspath(
    os.path.join(base_dir, "test_vae_training", "config_current_vae.tsv")
)

params = utils.read_config(config_filename)

dataset_name = params["dataset_name"]

normalized_compendium_filename = params["normalized_compendium_filename"]

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
train_vae_modules.train_vae(config_filename, normalized_compendium_filename)

# +
# Plot training and validation loss separately
stat_logs_filename = "logs/NN_2500_30/tybalt_2layer_30latent_stats.tsv"

stats = pd.read_csv(stat_logs_filename, sep="\t", index_col=None, header=0)
# -

plt.plot(stats["loss"])

plt.plot(stats["val_loss"], color="orange")

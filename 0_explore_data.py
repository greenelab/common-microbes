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

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from cm_modules import paths

# +
# Load dataset
# Each column is a bacterial taxon, and each row is a single sample.
# Rows are named using the format "${PROJECT_ID}_${SAMPLE_ID}"
# Each cell is a read count.
# There are ~11,800 samples in the preliminary dataset.
# sample x taxon

microbiome_data_filename = paths.RAW_MICROBIOME_DATA
microbiome_data = pd.read_csv(microbiome_data_filename, sep="\t", index_col=0, header=0)
# -

print(microbiome_data.shape)
microbiome_data.head()

# How sparse is this data matrix?
# How many 0s
num_zeros = (microbiome_data == 0).sum().sum()
total = microbiome_data.shape[0] * microbiome_data.shape[1]
num_zeros / total

# Number of 0's per taxa
zero_per_taxa = sns.displot((microbiome_data == 0).sum())
plt.show(zero_per_taxa)

# Plot distribution of abundances per taxa
boxplot = np.log10(1 + microbiome_data).boxplot(rot=90, fontsize=5)

plt.show(boxplot)
plt.set_title("log10 abundance per taxa")

# What is the range of values per feature (taxa)
microbiome_data.min()

# What is the range of values per feature (taxa)
microbiome_data.max()

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
#     display_name: Python [conda env:microbe]
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
# Dataset is of the form sample x taxon
# Each column is a bacterial taxon, and each row is a single sample.
# Rows are named using the format "${PROJECT_ID}_${SAMPLE_ID}"
# Each cell is a read count.
# There are ~11,800 samples in the preliminary dataset.

microbiome_data_filename = paths.RAW_MICROBIOME_DATA
microbiome_data = pd.read_csv(microbiome_data_filename, sep="\t", index_col=0, header=0)
# -

print(microbiome_data.shape)
microbiome_data.head()

# How sparse is this data matrix?
# How many 0s across the entire matrix?
num_zeros = (microbiome_data == 0).sum().sum()
total = microbiome_data.shape[0] * microbiome_data.shape[1]
num_zeros / total

# Number of 0's per taxon
zero_per_taxon = sns.histplot((microbiome_data == 0).sum())
plt.show(zero_per_taxon)
plt.ylabel("Count")
plt.xlabel("Number of 0's per taxon")
plt.title("Distribution of 0's per taxon")

# Plot distribution of abundances per taxon
boxplot = np.log10(1 + microbiome_data.sample(n=10, axis="columns")).boxplot(
    rot=90, fontsize=15
)

plt.show(boxplot)
plt.ylabel("log10 abundance")
plt.xlabel("Random set of 10 taxon")
plt.title("log10 abundance per taxon")

# **Takeaway**:
#
# Overall we can see that our input data is _very_ sparse. Most taxa only found in a small subset of samples. We will need to modify our VAE model to account for this sparsity.

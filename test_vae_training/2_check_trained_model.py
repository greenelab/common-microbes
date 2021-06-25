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

# # Check training model
#
# This notebook will examine the trained model to see if it learned patterns of the input data. Specifically this notebook will look at:
#
# 1. Is there any structure in the data or just noise?
# 2. Does our latent space capture the clusters in our input data? Are there samples that we know should cluster together? Do we find those in the input and encoded data?
# 3. Do simulated samples, looks similar to the input data?

# +
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
import os
import glob

# import umap
import pandas as pd
from keras.models import load_model
from sklearn.decomposition import PCA
from plotnine import (
    ggplot,
    labs,
    geom_line,
    geom_point,
    geom_errorbar,
    aes,
    ggsave,
    theme_bw,
    theme,
    xlim,
    ylim,
    facet_wrap,
    scale_color_manual,
    guides,
    guide_legend,
    element_blank,
    element_text,
    element_rect,
    element_line,
    coords,
)
from cm_modules import paths
from ponyo import utils, simulate_expression_data

random_state = 123

# +
base_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))

# Read in config variables
config_filename = os.path.abspath(
    os.path.join(base_dir, "test_vae_training", "config_current_vae.tsv")
)

params = utils.read_config(config_filename)

dataset_name = params["dataset_name"]
NN_architecture = params["NN_architecture"]
normalized_compendium_filename = params["normalized_compendium_filename"]

NN_dir = os.path.join(base_dir, dataset_name, "models", NN_architecture)
# -

# Import normalized data
normalized_compendium = pd.read_csv(
    normalized_compendium_filename, sep="\t", index_col=0, header=0
)

normalized_compendium.head()

# Drop outliers found manually by visual inspection
samples_to_drop = [
    "PRJEB34610_ERR3561806",
    "PRJEB34610_ERR3561830",
    "PRJNA297268_SRR2568180",
]
normalized_compendium = normalized_compendium.drop(samples_to_drop)

# ### Plot PCs of normalized data

pca = PCA(n_components=2)

# +
# Get and save PCA model
model = pca.fit(normalized_compendium)
# model = umap.UMAP(random_state=random_state).fit(normalized_compendium)

compendium_PCAencoded = model.transform(normalized_compendium)

compendium_PCAencoded_df = pd.DataFrame(
    data=compendium_PCAencoded, index=normalized_compendium.index, columns=["1", "2"]
)

# +
# Plot
fig = ggplot(compendium_PCAencoded_df, aes(x="1", y="2"))
fig += geom_point(alpha=0.2)
fig += labs(x="PCA 1", y="PCA 2", title="PCA normalized compendium")
fig += theme_bw()
fig += theme(
    legend_title_align="center",
    plot_background=element_rect(fill="white"),
    legend_key=element_rect(fill="white", colour="white"),
    legend_title=element_text(family="sans-serif", size=15),
    legend_text=element_text(family="sans-serif", size=12),
    plot_title=element_text(family="sans-serif", size=15),
    axis_text=element_text(family="sans-serif", size=12),
    axis_title=element_text(family="sans-serif", size=15),
)
fig += guides(colour=guide_legend(override_aes={"alpha": 1}))

print(fig)
# -

# ### Plot PCs of encoded normalized data

# +
# Load VAE models
model_encoder_file = glob.glob(os.path.join(NN_dir, "*_encoder_model.h5"))[0]

weights_encoder_file = glob.glob(os.path.join(NN_dir, "*_encoder_weights.h5"))[0]

model_decoder_file = glob.glob(os.path.join(NN_dir, "*_decoder_model.h5"))[0]

weights_decoder_file = glob.glob(os.path.join(NN_dir, "*_decoder_weights.h5"))[0]

# Load saved models
loaded_model = load_model(model_encoder_file)
loaded_decode_model = load_model(model_decoder_file)

loaded_model.load_weights(weights_encoder_file)
loaded_decode_model.load_weights(weights_decoder_file)

# +
# Encode normalized compendium into latent space
compendium_encoded = loaded_model.predict_on_batch(normalized_compendium)

compendium_encoded_df = pd.DataFrame(
    data=compendium_encoded, index=normalized_compendium.index
)

# +
# Get and save PCA model
model = pca.fit(compendium_encoded_df)
# model = umap.UMAP(random_state=random_state).fit(compendium_encoded_df)

latent_compendium_PCAencoded = model.transform(compendium_encoded_df)

latent_compendium_PCAencoded_df = pd.DataFrame(
    data=latent_compendium_PCAencoded,
    index=compendium_encoded_df.index,
    columns=["1", "2"],
)

# +
# Plot umap of encoded data
fig = ggplot(latent_compendium_PCAencoded_df, aes(x="1", y="2"))
fig += geom_point(alpha=0.2)
fig += labs(x="PCA 1", y="PCA 2", title="PCA encoded normalized compendium")
fig += theme_bw()
fig += theme(
    legend_title_align="center",
    plot_background=element_rect(fill="white"),
    legend_key=element_rect(fill="white", colour="white"),
    legend_title=element_text(family="sans-serif", size=15),
    legend_text=element_text(family="sans-serif", size=12),
    plot_title=element_text(family="sans-serif", size=15),
    axis_text=element_text(family="sans-serif", size=12),
    axis_title=element_text(family="sans-serif", size=15),
)
fig += guides(colour=guide_legend(override_aes={"alpha": 1}))

print(fig)
# -

# ## Simulate vs real data
#
# Here we want to make sure that the distribution of the simulated data looks similar to the real data -- i.e. the sparsity is similar.

simulated_data = simulate_expression_data.run_sample_simulation(
    loaded_model, loaded_decode_model, normalized_compendium, 5
)

simulated_data.head()

# How sparse is this data matrix?
# How many 0s across the entire matrix?
num_zeros = (simulated_data == 0).sum().sum()
total = simulated_data.shape[0] * simulated_data.shape[1]
num_zeros / total

# How lowly expressed is this data matrix?
num_low = (simulated_data < 0.1).sum().sum()
num_low / total

# **Takeaways:**
# * Latent representation captures the distribution of the input dataset, which is mainly composed of a single large cluster.
# * The latent space compresses the spreading seen in the input dataset, as previously seen
# * The simulated data, generated by randomly sampling from the latent space, is generating all lowly expressed microbes (i.e. all expression is below 0.1, but none are 0), which makes sense given the dominant signal in the data are these 0 counts. Since most genes have 0 counts, then the latent variables distribution will be skewed to 0 (though there is also a normality constraint in addition to the reconstruction error) so when the simulation goes to sample from the latent space that is why the values are so low.

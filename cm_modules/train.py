# Copyright 2016 Goekcen Eraslan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

from . import io
# from .hyper import hyper

import numpy as np
import tensorflow as tf
import keras.optimizers as opt
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import Iterator
# The below lines were added due to error: Cannot convert keras symbolic input
# to numpy array
# Found this stack overflow solution: https://stackoverflow.com/questions/65366442/cannot-convert-a-symbolic-keras-input-output-to-a-numpy-array-typeerror-when-usi
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


def train(adata, network, output_dir=None, optimizer='RMSprop', learning_rate=None,
          epochs=300, reduce_lr=10, output_subset=None, use_raw_as_output=True,
          early_stop=15, batch_size=32, clip_grad=5., save_weights=False,
          validation_split=0.1, tensorboard=False, verbose=True, threads=None,
          **kwds):

    tf.compat.v1.keras.backend.set_session(
        tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=threads,
                inter_op_parallelism_threads=threads,
            )
        )
    )
    model = network.model
    loss = network.loss

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if learning_rate is None:
        optimizer = opt.__dict__[optimizer](clipvalue=clip_grad)
    else:
        optimizer = opt.__dict__[optimizer](lr=learning_rate, clipvalue=clip_grad)
    print("about to compiile")

    # Instantiate an optimizer.
    # if optimizer == "Adam":
    #     optimizer = tf.keras.optimizers.Adam()

    print("optimizer", optimizer)
    model.compile(loss=loss, optimizer=optimizer, experimental_run_tf_function=False)

    # Callbacks
    callbacks = []

    if save_weights and output_dir is not None:
        checkpointer = ModelCheckpoint(filepath="%s/weights.hdf5" % output_dir,
                                       verbose=verbose,
                                       save_weights_only=True,
                                       save_best_only=True)
        callbacks.append(checkpointer)
    if reduce_lr:
        lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr, verbose=verbose)
        callbacks.append(lr_cb)
    if early_stop:
        es_cb = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=verbose)
        callbacks.append(es_cb)
    if tensorboard:
        tb_log_dir = os.path.join(output_dir, 'tb')
        tb_cb = TensorBoard(log_dir=tb_log_dir, histogram_freq=1, write_grads=True)
        callbacks.append(tb_cb)

    if verbose:
        model.summary()

    # The format of the input data is modified to use pandas dataframe instead
    # of scanpy object because getting the following error:
    #
    inputs = {'count': adata.X, 'size_factors': adata.obs.size_factors}
    print("inputs", inputs)

    # Make size_factors a vector of ones
    # size_factors = np.ones(adata.shape[0])
    # inputs = {'count': adata, 'size_factors': size_factors}

    if output_subset:
        gene_idx = [np.where(adata.raw.var_names == x)[0][0] for x in output_subset]
        output = adata.raw.X[:, gene_idx] if use_raw_as_output else adata.X[:, gene_idx]
    else:
        output = adata.raw.X if use_raw_as_output else adata.X
        # output = np.array(adata.iloc[1:,1:])
    print("output", output)
    loss = model.fit(inputs, output,
                     epochs=epochs,
                     batch_size=batch_size,
                     shuffle=True,
                     callbacks=callbacks,
                     validation_split=validation_split,
                     verbose=verbose,
                     **kwds)

    return

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Convolutional encoder with a per-subject 2D-Fourier channel merger; produces
# one embedding per keystroke window.
ENCODER = {
    "name": "SimpleConvTimeAgg",
    "time_agg_out": "att",
    "dropout_input": 0.2,
    "conv_dropout": 0.5,
    "hidden": 2048,
    "batch_norm": True,
    "depth": 8,
    "dilation_period": 3,
    "kernel_size": 3,
    "relu_leakiness": 0.01,
    "initial_linear": 512,
    "gelu": True,
    "skip": True,
    "scale": 0.1,
    # "subject_layers_config": {}, # this will append shit later
    "subject_layers_config": None, # setting it to None make it so there would be no subject specific layer in final model
    
    # "merger_config": {
    #     "n_virtual_channels": 270,
    #     "fourier_emb_config": {"n_freqs": None, "total_dim": 2048, "n_dims": 2},
    #     "dropout": 0.2,
    #     "usage_penalty": 1.0,
    #     "per_subject": True,
    #     "embed_ref": False,
    # },
    "merger_config": None,
}
"""
Merger : B, C, T --> B, O, T (O is 270)
So each output virtual channel's time series is a learned weighted average of the physical channels' time series, where the weights depend only on where the physical channels are located in space (not on the signal itself) — collapsing a variable, geometry-dependent set of C real electrodes down to a fixed, geometry-agnostic set of O virtual channels.
Only spatial features. The time dimension (T) never touches the weight computation at all.
"""

# Sentence-level transformer over the per-keystroke embeddings.
TRANSFORMER = {
    "name": "TransformerEncoder",
    "alibi_pos_bias": True,
    "depth": 4,
    "heads": 2,
}
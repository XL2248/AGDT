# coding=utf-8
# Code modified from Tensor2Tensor library
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import code

def create_inference_graph(model_fns, features, params):
    if not isinstance(model_fns, (list, tuple)):
        model_fns = [model_fns]

    decode_length = params.decode_length#1
    beam_size = params.beam_size#4
    top_beams = params.top_beams#1
    alpha = params.decode_alpha#0.6
    for i, model_fn in enumerate(model_fns):
            result = model_fn(features)

    return result

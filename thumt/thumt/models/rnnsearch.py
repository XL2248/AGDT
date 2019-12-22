# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.interface as interface
import thumt.data.vocab as vocabulary
import thumt.layers as layers
import code
import numpy as np
import os

def _copy_through(time, length, output, new_output):
    copy_cond = (time >= length)
    return tf.where(copy_cond, output, new_output)

def _process_vocabulary(filename):
    vocab = []
    with tf.gfile.GFile(filename) as fd:
        for line in fd:
            word = line.strip()
            vocab.append(word)

    return vocab

def _load_embedding(word_list, params, uniform_scale = 0.25, dimension_size = 300):

    word2embed = {}
    if params.embed_file == 'w2v':
        file_path = params.embedding_path
    else:
        file_path = params.embedding_path
    
    with open(file_path, 'r') as fopen:
        for line in fopen:
            w = line.strip().split()
            word2embed[' '.join(w[:-dimension_size])] = w[-dimension_size:]
    word_vectors = []
    
    c = 0
    for word in word_list:
        if word in word2embed:
            c += 1
            s = np.array(word2embed[word], dtype=np.float32)
            word_vectors.append(s)
        else:
            word_vectors.append(np.random.uniform(-uniform_scale, uniform_scale, dimension_size))

    return np.array(word_vectors, dtype=np.float32)

def _gru_encoder(cell, inputs, aspect_inputs, sequence_length, params, initial_state, dtype=None):
    # Assume that the underlying cell is GRUCell-like
    output_size = cell.output_size
    dtype = dtype or inputs.dtype

    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]

    zero_output = tf.zeros([batch, output_size], dtype)

    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    input_ta = tf.TensorArray(dtype, time_steps,
                              tensor_array_name="input_array")
    input_aspect = tf.TensorArray(dtype, time_steps,
                              tensor_array_name="input_array")
    output_ta = tf.TensorArray(dtype, time_steps,
                               tensor_array_name="output_array")
    input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))
    input_aspect = input_aspect.unstack(tf.transpose(aspect_inputs, [1, 0, 2]))

    inp_aspect_t = input_aspect.read(0)
    def loop_func(t, out_ta, state):
        inp_t = input_ta.read(t)
        
        cell_output, new_state = cell(inp_t, inp_aspect_t, params, state)
        cell_output = _copy_through(t, sequence_length, zero_output,
                                    cell_output)
        new_state = _copy_through(t, sequence_length, state, new_state)
        out_ta = out_ta.write(t, cell_output)
        return t + 1, out_ta, new_state

    time = tf.constant(0, dtype=tf.int32, name="time")
    loop_vars = (time, output_ta, initial_state)

    outputs = tf.while_loop(lambda t, *_: t < time_steps, loop_func,
                            loop_vars, parallel_iterations=32,
                            swap_memory=True)

    output_final_ta = outputs[1]
    final_state = outputs[2]

    all_output = output_final_ta.stack()
    all_output.set_shape([None, None, output_size])
    all_output = tf.transpose(all_output, [1, 0, 2])

    return all_output, final_state


def _encoder(cell_fw, cell_bw, inputs, aspect_inputs, sequence_length, params, dtype=None,
             scope=None):
    with tf.variable_scope(scope or "encoder",
                           values=[inputs, sequence_length, aspect_inputs]):
        aspect_w = aspect_inputs
        inputs_fw = inputs
        inputs_bw = tf.reverse_sequence(inputs, sequence_length,
                                        batch_axis=0, seq_axis=1)
        with tf.variable_scope("forward"):
            output_fw, state_fw = _gru_encoder(cell_fw, inputs_fw, aspect_w,
                                               sequence_length, params, None,
                                               dtype=dtype)
        with tf.variable_scope("backward"):
            output_bw, state_bw = _gru_encoder(cell_bw, inputs_bw, aspect_w,
                                               sequence_length, params, None,
                                               dtype=dtype)
            output_bw = tf.reverse_sequence(output_bw, sequence_length,
                                            batch_axis=0, seq_axis=1)
        results = {
            "annotation": tf.concat([output_fw, output_bw], axis=2),
            "outputs": {
                "forward": output_fw,
                "backward": output_bw
            },
            "final_states": {
                "forward": state_fw,
                "backward": state_bw
            }
        }

        return results


def _decoder(cell, inputs, memory, sequence_length, initial_state, dtype=None,
             scope=None):
    # Assume that the underlying cell is GRUCell-like
    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]
    dtype = dtype or inputs.dtype
    output_size = cell.output_size
    zero_output = tf.zeros([batch, output_size], dtype)
    zero_value = tf.zeros([batch, memory.shape[-1].value], dtype)

    with tf.variable_scope(scope or "decoder", dtype=dtype):
        inputs = tf.transpose(inputs, [1, 0, 2])
        memory = tf.transpose(memory, [1, 0, 2])
        '''
 mem_mask = tf.sequence_mask(sequence_length["source"],
                                    maxlen=tf.shape(memory)[1],
                                    dtype=tf.float32)
        bias = layers.attention.attention_bias(mem_mask, "masking")
        bias = tf.squeeze(bias, axis=[1])
        cache = layers.attention.attention_mhead(None, memory, None, output_size)
    '''
        input_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="input_array")
        memory_ta = tf.TensorArray(tf.float32, tf.shape(memory)[0],
                                   tensor_array_name="memory_array")
        output_ta = tf.TensorArray(tf.float32, time_steps,
                                   tensor_array_name="output_array")
    # '''
    #     value_ta = tf.TensorArray(tf.float32, time_steps,
    #                               tensor_array_name="value_array")
    #     alpha_ta = tf.TensorArray(tf.float32, time_steps,
    #                               tensor_array_name="alpha_array")
    # '''
        input_ta = input_ta.unstack(inputs)
        memory_ta = memory_ta.unstack(memory)
        initial_state = layers.nn.linear(initial_state, output_size, True,
                                         False, scope="s_transform")
        initial_state = tf.tanh(initial_state)

        def loop_func(t, out_ta, state):
            inp_t = input_ta.read(t)
            mem_t = memory_ta.read(t)
            '''
     output1, state1 = cell_cond(inp_t, state)
            state1 = _copy_through(t, sequence_length["target"], state,
                                   state1)
            results = layers.attention.attention_mhead(state1, memory, bias,
                                                       output_size,
                                                       cache={"key": cache_key})
            alpha = results["weight"]
            context = results["value"]
     '''
            cell_input = [inp_t, mem_t]
            cell_output, new_state = cell(cell_input, state)
            cell_output = _copy_through(t, sequence_length["target"],
                                        zero_output, cell_output)
            new_state = _copy_through(t, sequence_length["target"], state,
                                      new_state)
            '''
     new_value = _copy_through(t, sequence_length["target"], zero_value,
                                      context)
     '''
            out_ta = out_ta.write(t, cell_output)
            #att_ta = att_ta.write(t, alpha)
            #val_ta = val_ta.write(t, new_value)
            return t + 1, out_ta, new_state

        time = tf.constant(0, dtype=tf.int32, name="time")
        loop_vars = (time, output_ta, initial_state)

        outputs = tf.while_loop(lambda t, *_: t < time_steps,
                                loop_func, loop_vars,
                                parallel_iterations=32,
                                swap_memory=True)

        output_final_ta = outputs[1]
        #value_final_ta = outputs[3]

        final_output = output_final_ta.stack()
        final_output.set_shape([None, None, output_size])
        final_output = tf.transpose(final_output, [1, 0, 2])

# '''
#         final_value = value_final_ta.stack()
#         final_value.set_shape([None, None, memory.shape[-1].value])
#         final_value = tf.transpose(final_value, [1, 0, 2])
# '''
        result = {
            "outputs": final_output,
            "initial_state": initial_state
        }

    return result

def model_graph(features, labels, params):

    src_vocab_size = len(params.vocabulary["source"])+2
    aspect_vocab_size = len(params.vocabulary["aspect_word"])
    
    tgt_vocab_size = len(params.vocabulary["target"])

    with tf.variable_scope("source_embedding"):
        if params.use_vec:
            src_emb = tf.Variable(_load_embedding(params.vocabulary["source"], params), name="embedding", trainable=False)
        else:
            src_emb = tf.get_variable("embedding",
                                      [src_vocab_size, params.embedding_size],initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        src_bias = tf.get_variable("bias", [params.embedding_size])
        src_inputs = tf.nn.embedding_lookup(src_emb, features["source"])

    with tf.variable_scope("aspect_embedding"):
        if params.use_vec_a:
            aspect_emb = tf.Variable(_load_embedding(params.vocabulary["aspect_word"], params), name="embedding", trainable=False)
        else:
            aspect_emb = tf.get_variable("embedding",
                                      [aspect_vocab_size, params.embedding_size],initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        aspect_bias = tf.get_variable("bias", [params.embedding_size])
        aspect_inputs = tf.nn.embedding_lookup(aspect_emb, features["aspect_word"])


    src_inputs = tf.nn.bias_add(src_inputs, src_bias)
    aspect_inputs = tf.nn.bias_add(aspect_inputs, aspect_bias)


    src_inputs = layers.attention.add_timing_signal(src_inputs)
    aspect_inputs = layers.attention.add_timing_signal(aspect_inputs)

    if params.dropout and not params.use_variational_dropout:
        src_inputs = tf.nn.dropout(src_inputs, 1.0 - params.dropout)
        aspect_inputs = tf.nn.dropout(aspect_inputs, 1.0 - params.dropout)
    maxout_size = params.hidden_size // params.maxnum
    # encoder
    if params.rnn_cell == "DL4MTGRULAUTransiCell":#DT
        cell_fw = layers.rnn_cell.DL4MTGRULAUTransiLNCell(params.num_transi, params.hidden_size, 1.0 - params.rnn_dropout)
        cell_bw = layers.rnn_cell.DL4MTGRULAUTransiLNCell(params.num_transi, params.hidden_size, 1.0 - params.rnn_dropout)
    maxout_size = params.hidden_size // params.maxnum
    
    a = 0.0
    predic_loss = 0.0
    if params.task == "acsa":
        if params.use_prediction:
            a = params.alpha
            encoder_output = _encoder(cell_fw, cell_bw, src_inputs, aspect_inputs, features["source_length"], params)
            readout = layers.nn.maxout(encoder_output["annotation"], maxout_size, params.maxnum, concat=False, scope="maxout_size-aspect")
            readout = tf.tanh(readout)
            if params.dropout and not params.use_variational_dropout:
                readout = tf.nn.dropout(readout, 1.0 - params.dropout)
            logits_aspect = layers.nn.linear(tf.reduce_max(readout, axis = 1), len(params.vocabulary["aspect_word"]), True, False, scope="softmax-aspect")
            ce_aspect = layers.nn.smoothed_softmax_cross_entropy_with_logits(
                logits=logits_aspect,
                labels=features["aspect_word"],
                smoothing=params.label_smoothing,
                normalize=True
            )
            predic_loss = a * tf.reduce_mean(ce_aspect)
        else:
            encoder_output = _encoder(cell_fw, cell_bw, src_inputs, aspect_inputs, features["source_length"], params)
#end of prediction
        if params.use_aspect:
            dim1 = tf.shape(encoder_output["annotation"])[1]
            aspect = tf.tile(aspect_inputs, [1, dim1 , 1])
            fbw = tf.concat([encoder_output["annotation"],  aspect], -1)
            x1 = tf.nn.relu(fbw)
        else:
            x1 = tf.nn.relu(encoder_output["annotation"])
       
        if params.use_capsule_net:
            logit = layers.capsnet.rooting(input_tensor=x1,
                                                                       mask_weight=features["source_length"],
                                                                       num_cap=params.class_num, iters=3, output_size=16, output_transform=False,
                                                                       stop_grad=True,bias=True,dtype=None,caps_relu=False,scope="caps_rooting")
        else:
            x3 = tf.reduce_max(x1, axis = 1)
        
            readout1 = layers.nn.maxout(x3, maxout_size, params.maxnum,
                                       concat=False)
            readout1 = tf.tanh(readout1)
            logit = layers.nn.linear(readout1, tgt_vocab_size, True, False, scope="softmax")
        
        if labels is None:
            return logit
        ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
            logits=logit,
            labels=labels,
            smoothing=params.label_smoothing,
            normalize=True
        )
        return tf.reduce_mean(ce) + predic_loss
    else:
        # aspect term
        if params.use_prediction:
            a = params.alpha
            encoder_output = _encoder(cell_fw, cell_bw, src_inputs, tf.expand_dims(tf.reduce_mean(aspect_inputs, axis = 1),axis=1), features["source_length"], params)
            readout = layers.nn.maxout(encoder_output["annotation"], maxout_size, params.maxnum, concat=False, scope="maxout_size-aspect")
            readout = tf.tanh(readout)
            if params.dropout and not params.use_variational_dropout:
                 readout = tf.nn.dropout(readout, 1.0 - params.dropout)
            logits_aspect = layers.nn.linear(tf.reduce_max(readout, axis = 1), len(params.vocabulary["aspect_word"]), True, False, scope="softmax-aspect")
            ce_aspect = layers.nn.smoothed_sigmoid_cross_entropy_with_logits(# multi label prediction
                logits=logits_aspect,
                labels=features["aspect_word"],
                tes=features["aspect_word"]
              )
            predic_loss = a * tf.reduce_mean(ce_aspect)
        else:
         	 encoder_output = _encoder(cell_fw, cell_bw, src_inputs, tf.expand_dims(tf.reduce_mean(aspect_inputs, axis = 1),axis=1), features["source_length"], params)
         #end of prediction
        if params.use_aspect:
            dim1 = tf.shape(encoder_output["annotation"])[1]
            aspect = tf.tile(tf.expand_dims(tf.reduce_mean(aspect_inputs, axis = 1),  axis=1), [1, dim1 , 1])
            fbw = tf.concat([encoder_output["annotation"] ,  aspect], -1)
            x1 = tf.nn.relu(fbw)
        else:
            x1 = encoder_output["annotation"]
        x1 = tf.nn.relu(x1)
        if params.use_capsule_net:
            logit = layers.capsnet.rooting(input_tensor=x1,
                                                                       mask_weight=features["source_length"],
                                                                       num_cap=params.class_num, iters=3, output_size=16, output_transform=False,
                                                                       stop_grad=True,bias=True,dtype=None,caps_relu=False,scope="caps_rooting")
        else:
            x3 = tf.reduce_max(x1, axis = 1)
        
            readout1 = layers.nn.maxout(x3, maxout_size, params.maxnum,
                                       concat=False)
            readout1 = tf.tanh(readout1)
            logit = layers.nn.linear(readout1, tgt_vocab_size, True, False, scope="softmax")
        if labels is None:
            return logit
        ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
            logits=logit,
            labels=labels,
            smoothing=params.label_smoothing,
            normalize=True
        )
        return tf.reduce_mean(ce) + predic_loss

class RNNsearch(interface.NMTModel):
    def __init__(self, params, scope="rnnsearch"):
        super(RNNsearch, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer):
        def training_fn(features, params=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=tf.AUTO_REUSE):
                loss = model_graph(features, features["target"], params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
            params.dropout = 0.0
            params.rnn_dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                logits = model_graph(features, None, params)

            return logits

        return evaluation_fn

    def get_inference_func(self):
        def inference_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
            params.dropout = 0.0
            params.rnn_dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                logits = model_graph(features, None, params)

            return logits

        return inference_fn

    @staticmethod
    def get_name():
        return "rnnsearch"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # vocabulary
            pad="<pad>",
            bos="<bos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            # model
            rnn_cell="DL4MTGRULAUTransiCell",
            embedding_size=256,
            hidden_size=300,
            maxnum=1,
            # regularization
            dropout=0.5,
            rnn_dropout=0.3,
            use_variational_dropout=False,
            label_smoothing=0.1,
            constant_batch_size=True,
            batch_size=128,
            max_length=100,
            clip_grad_norm=5.0,
            loss=0.0
        )

        return params

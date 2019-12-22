# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import operator
import code
import re
import random
import numpy as np
import tensorflow as tf
import functools
from pathlib import Path
from sacremoses import MosesTokenizer

def batch_examples(example, batch_size, max_length, mantissa_bits,
                   shard_multiplier=1, length_multiplier=1, constant=False,
                   num_threads=4, drop_long_sequences=True):
    """ Batch examples

    :param example: A dictionary of <feature name, Tensor>.
    :param batch_size: The number of tokens or sentences in a batch
    :param max_length: The maximum length of a example to keep
    :param mantissa_bits: An integer
    :param shard_multiplier: an integer increasing the batch_size to suit
        splitting across data shards.
    :param length_multiplier: an integer multiplier that is used to
        increase the batch sizes and sequence length tolerance.
    :param constant: Whether to use constant batch size
    :param num_threads: Number of threads
    :param drop_long_sequences: Whether to drop long sequences

    :returns: A dictionary of batched examples
    """

    with tf.name_scope("batch_examples"):
        max_length = max_length or batch_size
        min_length = 8
        mantissa_bits = mantissa_bits

        # Compute boundaries
        x = min_length
        boundaries = []

        while x < max_length:
            boundaries.append(x)
            x += 2 ** max(0, int(math.log(x, 2)) - mantissa_bits)

        # Whether the batch size is constant
        if not constant:
            batch_sizes = [max(1, batch_size // length)
                           for length in boundaries + [max_length]]
            batch_sizes = [b * shard_multiplier for b in batch_sizes]
            bucket_capacities = [2 * b for b in batch_sizes]
        else:
            batch_sizes = batch_size * shard_multiplier
            bucket_capacities = [2 * n for n in boundaries + [max_length]]

        max_length *= length_multiplier
        boundaries = [boundary * length_multiplier for boundary in boundaries]
        max_length = max_length if drop_long_sequences else 10 ** 9

        # The queue to bucket on will be chosen based on maximum length
        max_example_length = 0
        for v in example.values():
            if v.shape.ndims > 0:
                seq_length = tf.shape(v)[0]
                max_example_length = tf.maximum(max_example_length, seq_length)

        (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
            max_example_length,
            example,
            batch_sizes,
            [b + 1 for b in boundaries],
            num_threads=num_threads,
            capacity=2,  # Number of full batches to store, we don't need many.
            bucket_capacities=bucket_capacities,
            dynamic_pad=True,
            keep_input=(max_example_length <= max_length)
        )

    return outputs

def English(text):
    cleaned = re.findall('[a-zA-Z0-9\']+',text)
    cleaned = ' '.join(cleaned)  
    cleaned = ' '.join(cleaned.split())                    
    return cleaned
    
def get_training_input(filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filename, target_filename]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """
    tk = MosesTokenizer()
    with tf.device("/cpu:0"):

        def generator(words):
            
            target_set = set()
            with Path(words).open('r') as f_words:
                flag = 0
                for line in f_words:
                    flag += 1
                    sentence, aspect, sentiment = line.split('\t')
                    sentence = [word.encode() for word in tk.tokenize(sentence)]
                    aspect_word = [word.encode() for word in tk.tokenize(aspect)]
                    sentiment = [English(sentiment).encode()]
                    chars = [[c.encode() for c in w] for w in sentence]
                    assert len(chars) == len(sentence)
                    char_lengths = [len(c) for c in chars]
                    chars = [c + [params.pad.encode()] * (max(char_lengths) - l) for c, l in zip(chars, char_lengths)]

                    yield ((sentence, aspect_word, chars, char_lengths), sentiment)
            params.train_rows = flag

        shapes = (([None], [None], [None, None], [None]), 
                  [None])           

        types = ((tf.string, tf.string, tf.string, tf.int32),
                 tf.string)
        dataset = tf.data.Dataset.from_generator(
            functools.partial(
                generator,
                filenames[0]
            ),
            output_shapes=shapes,
            output_types=types
        )
        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()
        dataset = dataset.map(
            lambda src, tgt: {
                "source": src[0],
                "aspect_word": src[1],
                "chars": src[2],
                "char_length": src[3],
                "target": tgt,
                "source_length": tf.shape(src[0])[0],
                "aspect_word_length": tf.shape(src[1])[0],
                "target_length": tf.shape(tgt)[0]
            },
            num_parallel_calls=params.num_threads
        )
    

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        aspect_word_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["aspect_word"]),
            default_value=params.mapping["aspect_word"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=0
        )
        char_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["char"]),
            default_value=params.mapping["char"][params.unk]
        )

        # String to index lookup
        features["source"] = src_table.lookup(features["source"])
        features["aspect_word"] = aspect_word_table.lookup(features["aspect_word"])
        features["target"] = tgt_table.lookup(features["target"])
        features["chars"] = char_table.lookup(features["chars"])


        # Batching
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=len(params.device_list),
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # Convert to int32
        features["source"] = tf.to_int32(features["source"])
        features["aspect_word"] = tf.to_int32(features["aspect_word"])
        features["target"] = tf.to_int32(features["target"])
        features["source_length"] = tf.to_int32(features["source_length"])
        features["target_length"] = tf.to_int32(features["target_length"])
        features["char_length"] = tf.to_int32(features["char_length"])

        return features


def sort_input_file(filename, reverse=True):
    # Read file
    with tf.gfile.Open(filename) as fd:
        inputs = [line.strip() for line in fd]

    input_lens = [
        (i, len(line.strip().split())) for i, line in enumerate(inputs)
    ]

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i

    return sorted_keys, sorted_inputs


def sort_and_zip_files(names):
    inputs = []
    input_lens = []
    files = [tf.gfile.GFile(name) for name in names]

    count = 0

    for lines in zip(*files):
        lines = [line.strip() for line in lines]
        input_lens.append((count, len(lines[0].split())))
        inputs.append(lines)
        count += 1

    # Close files
    for fd in files:
        fd.close()

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=True)
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])

    return [list(x) for x in zip(*sorted_inputs)]

def get_evaluation_input(filename, params):
    tk = MosesTokenizer()
    with tf.device("/cpu:0"):
        def generator(words):
            with Path(words).open('r') as f_words:
                flag1 = 0
                for line in f_words:
                    flag1 += 1
                    sentence, aspect, flag, sentiment = line.split('\t')
                    sentence = [word.encode() for word in tk.tokenize(sentence)]                 
                    aspect_word = [word.encode() for word in tk.tokenize(aspect)]
                    sentiment = [English(sentiment).encode()]
                    flag = [English(flag).encode()]

                    chars = [[c.encode() for c in w] for w in sentence]
                    char_lengths = [len(c) for c in chars]
                    chars = [c + [params.pad.encode()] * (max(char_lengths) - l) for c, l in zip(chars, char_lengths)]
                    
                    assert len(chars) == len(sentence)
                    yield ((sentence, flag, aspect_word, chars, char_lengths), sentiment)
            params.test_rows = flag1
        shapes = (([None], [None], [None], [None, None], [None]), [None])           

        types = ((tf.string, tf.string, tf.string, tf.string, tf.int32), tf.string )

        dataset = tf.data.Dataset.from_generator(
            functools.partial(
                generator,
                filename[0]
            ),
            output_shapes=shapes,
            output_types=types
        )
        dataset = dataset.map(
            lambda src, tgt: {
                "source": src[0],
                "flag": src[1],
                "aspect_word": src[2],
                "chars": src[3],
                "char_length": src[4],
                "source_length": tf.shape(src[0])[0],
                "target": tgt
            },
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.padded_batch(
            params.eval_batch_size,
            {
                "source": [None],
                "flag": [None],
                "aspect_word": [None],
                "chars": [None, None],
                "char_length": [None],
                "source_length": [],
                "target": [None]
            },
            {
                "source": params.pad,
                "flag": params.pad,
                "aspect_word": params.pad,
                "chars": params.pad,
                "char_length": 0,
                "source_length": 0,
                "target": params.pad
            }
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value = params.mapping["source"][params.unk]
        )
        
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value = -100
        )
        flag_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["flag"])
        )
        aspect_word_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["aspect_word"]),
            default_value = params.mapping["aspect_word"][params.unk]
        )

        char_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["char"]),
            default_value=params.mapping["char"][params.unk]
        )


        features["source"] = src_table.lookup(features["source"])
        features["aspect_word"] = aspect_word_table.lookup(features["aspect_word"])
        features["target"] = tgt_table.lookup(features["target"])
        features["flag"] = flag_table.lookup(features["flag"])
        features["chars"] = char_table.lookup(features["chars"])


        features["source"] = tf.to_int32(features["source"])
        features["aspect_word"] = tf.to_int32(features["aspect_word"])
        features["flag"] = tf.to_int32(features["flag"])
        features["source_length"] = tf.to_int32(features["source_length"])
        features["char_length"] = tf.to_int32(features["char_length"])


    return features

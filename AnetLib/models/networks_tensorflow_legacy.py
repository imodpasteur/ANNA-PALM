from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import json
import glob
import random
import collections
import math
import time
import threading

EPS = 1e-12

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_loss_L2,  gen_grads_and_vars, train")


def preprocess(images, mode='min_max[0,1]'):
    with tf.name_scope("preprocess"):
        if mode == 'mean_std':
            # NOTE: we still use "*2-1" scale for lagacy models, it should be removed later
            return tf.map_fn(lambda image: tf.image.per_image_standardization(image), images) * 2 - 1
        elif mode == 'min_max[0,1]':
            images = tf.clip_by_value(images, 0, 255.0) / 255.0
            # [0, 1] => [-1, 1]
            return images * 2 - 1
        else:
            raise NotImplemented


def deprocess(images, mode='min_max[0,1]'):
    with tf.name_scope("deprocess"):
        if mode == 'min_max[0,1]':
            # [-1, 1] => [0, 1]
            images = (images + 1) / 2
            return tf.clip_by_value(images, 0, 1.0)
        else:
            raise NotImplemented


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv


def generate_unet(generator_inputs, generator_outputs_channels, ngf=64, bayesian_dropout=False, dropout_prob=0.5, output_num=1, activation=tf.tanh, use_resize_conv=False):
    assert bayesian_dropout == False and use_resize_conv == False and output_num == 1
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, ngf, stride=2)
        layers.append(output)

    layer_specs = [
        ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob= 1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def generate_discriminator(discrim_inputs, discrim_targets, ndf=64, discriminator_layer_num=3, no_lsgan=False):
    n_layers = discriminator_layer_num # default value 3
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(input, ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)
        # print('conv '+ str(ndf))

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            if i>3:
                out_channels /= min(2**(i-3), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)
            # print('conv '+ str(out_channels))

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        if not no_lsgan:
            output = convolved
        else:
            output = tf.sigmoid(convolved)
        layers.append(output)
        # print('conv '+ str(1))

    return layers[-1]


def create_model(inputs, targets, ngf=64, ndf=64, discriminator_layer_num=3, output_uncertainty=False, bayesian_dropout=False, use_resize_conv=False, no_lsgan=False, dropout_prob=0.5, gan_weight=1.0, l1_weight=100.0, l2_weight=0.0, lr=0.0002, beta1=0.5):
    assert bayesian_dropout == False and use_resize_conv == False
    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = generate_unet(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = generate_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = generate_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        if not no_lsgan:
            discrim_loss = tf.reduce_mean(tf.square(predict_real - 1)) + tf.reduce_mean(tf.square(predict_fake))
        else:
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        if not no_lsgan:
            gen_loss_GAN = tf.reduce_mean(tf.square(predict_fake-1))
        else:
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss_L2 = tf.reduce_mean(tf.square(targets - outputs))
        if l2_weight != 0:
            gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight + gen_loss_L2 * l2_weight
        else:
            gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    if l2_weight != 0:
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1, gen_loss_L2])
    else:
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_loss_L2=gen_loss_L2 if l2_weight == 0 else ema.average(gen_loss_L2),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def build_network(model_type, input_size, input_nc, output_nc, batch_size, use_resize_conv=False, norm_A=None, norm_B=None, include_summary=True, gan_weight=1.0, l1_weight=100.0, lr=0.0002, beta1=0.5):
    assert use_resize_conv==False and include_summary==True
    if norm_A is None:
        norm_A = 'mean_std'
    if norm_B is None:
        norm_B = 'min_max[0,1]'
    assert norm_A == 'mean_std' and norm_B == 'min_max[0,1]'

    print('INFO: building a legacy pix2pix network...')
    queue_path = tf.placeholder(tf.string, shape=[None, 1], name='queue_path')
    queue_input = tf.placeholder(tf.float32, shape=[None, input_size, input_size, input_nc], name='queue_input')
    queue_target = tf.placeholder(tf.float32, shape=[None, input_size, input_size, output_nc], name='queue_target')

    # Build an FIFOQueue
    queue = tf.FIFOQueue(capacity=50, dtypes=[tf.string, tf.float32, tf.float32], shapes=[[1], [input_size, input_size, input_nc], [input_size, input_size, output_nc]])
    enqueue_op = queue.enqueue_many([queue_path, queue_input, queue_target])
    dequeue_op = queue.dequeue()
    close_queue_op = queue.close(cancel_pending_enqueues=True)

    # tensorflow recommendation:
    # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
    paths_batch, raw_inputs_batch, raw_targets_batch = tf.train.batch(dequeue_op, batch_size=batch_size, capacity=40)
    inputs_batch = preprocess(raw_inputs_batch, mode='mean_std')
    targets_batch = preprocess(raw_targets_batch, mode='min_max[0,1]')

    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    # inputs and targets are [batch_size, height, width, channels]
    if model_type == 'unet_legacy':
        model = create_model(inputs_batch, targets_batch, dropout_prob=dropout_prob, gan_weight=0.0, l1_weight=0.0, l2_weight=l1_weight, bayesian_dropout=False, use_resize_conv=False, lr=lr, beta1=beta1)
    elif model_type == 'pix2pix_legacy':
        model = create_model(inputs_batch, targets_batch, dropout_prob=dropout_prob, gan_weight=gan_weight, l1_weight=l1_weight, l2_weight=0.0, bayesian_dropout=False, use_resize_conv=False, lr=lr, beta1=beta1)
    else:
        raise Exception('unsupported model')

    inputs = raw_inputs_batch
    targets = raw_targets_batch
    outputs = deprocess(model.outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": paths_batch,
            "inputs": inputs, # tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": targets, # tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": outputs, # tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }
        # if model.uncertainty is not None:
        #     display_fetches["aleatoric_uncertainty"] = model.uncertainty # tf.map_fn(tf.image.encode_png, converted_uncertainty, dtype=tf.string, name="uncertainty_png")

    if include_summary:
        def convert(image, scale=False):
            channel_count =  image.get_shape()[3]
            #if opt.aspect_ratio != 1.0:
                # upscale to correct aspect ratio
            #    size = [opt.input_size, int(round(opt.input_size * opt.aspect_ratio))]
            #    image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)
            if channel_count > 3:
                image = image[:, :, :, 0:3]
            elif channel_count == 2:
                image = tf.concat([image, tf.zeros([batch_size, input_size, input_size, 1])], axis=3)
            if scale:
                image = tf.div( tf.subtract(
                                  image,
                                  tf.reduce_min(image)
                               ),tf.subtract(
                                  tf.reduce_max(image),
                                  tf.reduce_min(image)
                               ))*255.0
            return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

        # reverse any processing on images so they can be written to disk or displayed to user
        with tf.name_scope("convert_inputs"):
            converted_inputs = convert(inputs)

        with tf.name_scope("convert_targets"):
            converted_targets = convert(targets)

        with tf.name_scope("convert_outputs"):
            converted_outputs = convert(outputs)

        # if model.uncertainty is not None:
        #     with tf.name_scope("convert_uncertainty"):
        #         converted_uncertainty = convert(model.uncertainty, scale=True)
        # else:
        #     converted_uncertainty = None

        with tf.name_scope("loss_fetches"):
            loss_fetches = {}
            loss_fetches["discrim_loss"] = model.discrim_loss
            loss_fetches["gen_loss_GAN"] = model.gen_loss_GAN
            loss_fetches["gen_loss_L1"] = model.gen_loss_L1
            loss_fetches["gen_loss_L2"] = model.gen_loss_L2

        # summaries
        with tf.name_scope("inputs_summary"):
            tf.summary.image("inputs", converted_inputs)

        with tf.name_scope("targets_summary"):
            tf.summary.image("targets", converted_targets)

        with tf.name_scope("outputs_summary"):
            tf.summary.image("outputs", converted_outputs)

        # if converted_uncertainty is not None:
        #     with tf.name_scope("uncertainty_summary"):
        #         tf.summary.image("uncertainty", converted_uncertainty)


        with tf.name_scope("predict_real_summary"):
            tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

        with tf.name_scope("predict_fake_summary"):
            tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

        tf.summary.scalar("discriminator_loss", model.discrim_loss)
        tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
        tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
        tf.summary.scalar("generator_loss_L2", model.gen_loss_L2)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)

        for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad)

        summary_merged = tf.summary.merge_all()
    else:
        summary_merged = None
    return model, (enqueue_op, dequeue_op, close_queue_op), display_fetches, loss_fetches, summary_merged

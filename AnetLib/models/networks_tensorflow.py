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
import scipy
from PIL import Image
import time
from .revnet import ReversibleNet
# from selu_utils import selu, dropout_selu

EPS = 1e-12
LR_RANGE_LIM = 50.0 # the max for the low-res input image should be 50 at least
TARGET_RANGE_LIM = 1.0 # the max for target image should be 5 at least
INPUT_SWITCH_POS = 0
CTRL_CHANNEL_POS = 1
Model = collections.namedtuple("Model", "type, outputs, targets, uncertainty, predict_real, predict_fake, inputs, lr_inputs, lr_predict_real, lr_predict_fake, squirrel_error_map, squirrel_discrim_loss, squirrel_discrim_grads_and_vars, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss, gen_loss_L1, gen_loss_L2, gen_loss_SSIM, gen_loss_squirrel, losses, gen_grads_and_vars, squirrel_discrim_train, train")

def tf_scale(image, range_lim=1):
    mn = tf.reduce_min(image, axis=(0, 1, 2), keepdims=True)
    mx = tf.reduce_max(image, axis=(0, 1, 2), keepdims=True)
    return (image - mn) / tf.maximum(range_lim, (mx - mn))

def preprocess(images, mode='min_max[0,1]', range_lim=1.0):
    if images is None:
        return None
    with tf.name_scope("preprocess"):
        if mode == 'mean_std':
            # NOTE: we still use "*2-1" scale for lagacy models, it should be able to work without
            return tf.map_fn(lambda image: tf.image.per_image_standardization(image), images) * 2 - 1
        elif mode == 'min_max[0,1]':
            # [0, 1] => [-1, 1]
            return tf_scale(images, range_lim=range_lim) * 2 - 1
        elif mode == 'scale[8bit]':
            return images / 255 * 2 -1
        elif mode == 'scale[16bit]':
            return images / 65535 * 2 - 1
        else:
            raise NotImplemented


def deprocess(images, mode='min_max[0,1]'):
    if images is None:
        return None
    with tf.name_scope("deprocess"):
        if mode == 'min_max[0,1]':
            # [-1, 1] => [0, 1]
            images = (images + 1) / 2
            return images
        elif mode == 'scale[8bit]':
            images = (images + 1) / 2
            return images * 255
        elif mode == 'scale[16bit]':
            images = (images + 1) / 2
            return images * 65535
        else:
            raise NotImplemented


def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

def conv7x7(batch_input, out_channels, stride):
    with tf.variable_scope("conv7x7"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [7, 7, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [3, 3], [3, 3], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

def conv3x3(batch_input):
    with tf.variable_scope("conv3x3"):
        in_channels = batch_input.get_shape()[3]
        out_channels = in_channels
        filter = tf.get_variable("filter", [3, 3, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        padded_in_1 = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        out_1 = tf.nn.conv2d(padded_in_1, filter, [1, 1, 1, 1], padding="VALID")
        return out_1

def conv1x1(batch_input, out_channels):
    with tf.variable_scope("conv1x1"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [1, 1, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(batch_input, filter, [1, 1, 1, 1], padding="VALID")
        return conv


def conv1x1b(batch_input, out_channels):
    with tf.variable_scope("conv1x1"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [1, 1, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        bias = tf.get_variable("bias", [out_channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(batch_input, filter, [1, 1, 1, 1], padding="VALID")
        conv = tf.nn.bias_add(conv, bias)
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

# this is a simpler version of Tensorflow's 'official' version. See:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
def batchnorm2(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv


def resizeconv(batch_input, out_channels, scale_factor=2):
    with tf.variable_scope("resizeconv"):
        batch, in_height, in_width, in_channels = batch_input.get_shape().as_list()
        batch_input = tf.image.resize_images(batch_input, size=(in_height*scale_factor, in_width*scale_factor), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        filter = tf.get_variable("filter", [3, 3, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, 1, 1, 1], padding="VALID")
        return conv


def total_variation_regularization(images):
    width_var = tf.nn.l2_loss(images[:,:-1,:,:] - images[:,1:,:,:])
    height_var = tf.nn.l2_loss(images[:,:,:-1,:] - images[:,:,1:,:])
    return tf.add(width_var, height_var)


def _tf_fspecial_gauss(size, sigma, channels=1):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))

    window = g / tf.reduce_sum(g)
    return tf.tile(window, (1,1,channels,channels))


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, filter_size=11, filter_sigma=1.5):
    _, height, width, ch = img1.get_shape().as_list()
    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0

    window = _tf_fspecial_gauss(size, sigma, ch) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2

    padded_img1 = tf.pad(img1, [[0, 0], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")
    padded_img2 = tf.pad(img2, [[0, 0], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")
    mu1 = tf.nn.conv2d(padded_img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(padded_img2, window, strides=[1,1,1,1], padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2

    paddedimg11 = padded_img1*padded_img1
    paddedimg22 = padded_img2*padded_img2
    paddedimg12 = padded_img1*padded_img2

    sigma1_sq = tf.nn.conv2d(paddedimg11, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(paddedimg22, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(paddedimg12, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    ssim_value = tf.clip_by_value(((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)), 0, 1)
    if cs_map:
        cs_map_value = tf.clip_by_value((2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2), 0, 1)
        value = (ssim_value, cs_map_value)
    else:
        value = ssim_value
    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim_resize(img1, img2, weights=None, return_ssim_map=None, filter_size=11, filter_sigma=1.5):
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    level = len(weights)
    assert return_ssim_map is None or return_ssim_map < level
    weight = tf.constant(weights, dtype=tf.float32)
    mssim = []
    mcs = []
    _, h, w, _ = img1.get_shape().as_list()
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False, filter_size=filter_size, filter_sigma=filter_sigma)
        if return_ssim_map == l:
            return_ssim_map = tf.image.resize_images(ssim_map, size=(h, w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        img1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        img2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*(mssim[level-1]**weight[level-1])
    if return_ssim_map is not None:
        return value, return_ssim_map
    else:
        return value


def tf_ms_ssim(img1, img2, weights=None, mean_metric=False):
    if weights is None:
        weights = [1, 1, 1, 1, 1] # [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] #[1, 1, 1, 1, 1] #
    level = len(weights)
    sigmas = [0.5]
    for i in range(level-1):
        sigmas.append(sigmas[-1]*2)
    weight = tf.constant(weights, dtype=tf.float32)
    mssim = []
    mcs = []
    for l, sigma in enumerate(sigmas):
        filter_size = int(max(sigma*4+1, 11))
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False, filter_size=filter_size, filter_sigma=sigma)
        mssim.append(ssim_map)
        mcs.append(cs_map)
    # list to tensor of dim D+1
    value = mssim[level-1]**weight[level-1]
    for l in range(level):
        value = value * (mcs[l]**weight[l])
    if mean_metric:
        return tf.reduce_mean(value)
    else:
        return value


def tf_gauss_conv(img, filter_size=11, filter_sigma=1.5):
    _, height, width, ch = img.get_shape().as_list()
    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0
    window = _tf_fspecial_gauss(size, sigma, ch) # window shape [size, size]
    padded_img = tf.pad(img, [[0, 0], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")
    return tf.nn.conv2d(padded_img, window, strides=[1,1,1,1], padding='VALID')

def tf_gauss_weighted_l1(img1, img2, mean_metric=True, filter_size=11, filter_sigma=1.5):
    diff = tf.abs(img1 - img2)
    l1 = tf_gauss_conv(diff, filter_size=filter_size, filter_sigma=filter_sigma)
    if mean_metric:
        return tf.reduce_mean(l1)
    else:
        return l1

def tf_ssim_l1_loss(img1, img2, mean_metric=True, filter_size=11, filter_sigma=1.5, alpha=0.84):
    L1 = tf_gauss_weighted_l1(img1, img2, mean_metric=False, filter_size=filter_size, filter_sigma=filter_sigma)
    if mean_metric:
        loss_ssim= 1 - tf_ssim(img1, img2, cs_map=False, mean_metric=True, filter_size=filter_size, filter_sigma=filter_sigma)
        loss_L1 = tf.reduce_mean(L1)
        value = loss_ssim * alpha + loss_L1 * (1-alpha)
    else:
        loss_ssim= 1 - tf_ssim(img1, img2, cs_map=False, mean_metric=False, filter_size=filter_size, filter_sigma=filter_sigma)
        value = loss_ssim * alpha + L1 * (1-alpha)

    return value, loss_ssim

def tf_ms_ssim_l1_loss(img1, img2, mean_metric=True, alpha=0.84):
    ms_ssim_map = tf_ms_ssim(img1, img2, mean_metric=False)
    l1_map = tf_gauss_weighted_l1(img1, img2, mean_metric=False, filter_size=33, filter_sigma=8.0)
    loss_map = (1-ms_ssim_map) * alpha + l1_map * (1-alpha)
    if mean_metric:
        return tf.reduce_mean(loss_map)
    else:
        return loss_map

def tf_l1_loss(img1, img2):
    diff = tf.abs(img1 - img2)
    return tf.reduce_mean(diff)

def tf_l2_loss(img1, img2):
    diff = tf.square(img1 - img2)
    return tf.reduce_mean(diff)

def generate_revgan_x_encoder(generator_inputs, ngf=64, lr_inputs=None, lr_pos=0):
    layers = []
    print("x_encoder")
    print(generator_inputs.shape)
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 256, 256, ngf]
    with tf.variable_scope("x_encoder_1"):
        if lr_inputs is not None and lr_pos == 0:
            generator_inputs = tf.concat([generator_inputs, lr_inputs], axis=3)
        convolved = conv7x7(generator_inputs, ngf, stride=1)
        output = batchnorm(convolved)
        output = lrelu(output, 0.2)
        if lr_inputs is not None and lr_pos == 1:
            layers.append(tf.concat([lr_inputs, output], axis=3))
        else:
            layers.append(output)
    print(output.shape)
    layer_specs = [
        ngf * 2,  # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
        ngf * 4,  # encoder_3: [batch, 128, 128, ngf * 2] => [batch, 64, 64, ngf * 4]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("x_encoder_%d" % (len(layers) + 1)):

            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(layers[-1], out_channels, stride=2)
            output = batchnorm(convolved)
            output = lrelu(output, 0.2)
            if lr_inputs is not None and lr_pos == len(layers) + 1:
                layers.append(tf.concat([lr_inputs, output], axis=3))
            else:
                layers.append(output)
        print(output.shape)
    return output

def generate_revgan_y_encoder(generator_inputs, ngf=64, lr_inputs=None, lr_pos=0):
    layers = []
    print("y_encoder")
    print(generator_inputs.shape)
    # encoder_1: [batch, 512, 512, in_channels] => [batch, 512, 512, ngf]
    with tf.variable_scope("y_encoder_1"):
        if lr_inputs is not None and lr_pos == 0:
            generator_inputs = tf.concat([generator_inputs, lr_inputs], axis=3)
        convolved = conv7x7(generator_inputs, ngf, stride=1)
        output = batchnorm(convolved)
        output = lrelu(output, 0.2)
        if lr_inputs is not None and lr_pos == 1:
            layers.append(tf.concat([lr_inputs, output], axis=3))
        else:
            layers.append(output)
    print(output.shape)
    layer_specs = [
        ngf * 2,  # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
        ngf * 4,  # encoder_3: [batch, 128, 128, ngf * 2] => [batch, 64, 64, ngf * 4]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("y_encoder_%d" % (len(layers) + 1)):

            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(layers[-1], out_channels, stride=2)
            output = batchnorm(convolved)
            output = lrelu(output, 0.2)
            if lr_inputs is not None and lr_pos == len(layers) + 1:
                layers.append(tf.concat([lr_inputs, output], axis=3))
            else:
                layers.append(output)
        print(output.shape)
    return output


def generate_revgan_x_decoder(generator_inputs, generator_outputs_channels, ngf=64, output_num=1,
                              activation=tf.tanh, use_resize_conv=False, lr_nc=0, lr_pos=0):

    print("x_decoder")
    print(generator_inputs.shape)
    layers = []
    layers.append(generator_inputs)

    layer_specs = [
        (ngf * 2, None),  # decoder_3: [batch, 128, 128, ngf * 4] => [batch, 256, 256, ngf * 2]
        (ngf, None),  # decoder_2: [batch, 256, 256, ngf * 2] => [batch, 512, 512, ngf]
    ]

    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        with tf.variable_scope("x_decoder_%d" % (decoder_layer + 1)):

            input = layers[-1]

            rectified = tf.nn.relu(input)

            if lr_nc > 0 and lr_pos == len(layer_specs) - decoder_layer:
                lr_output = conv7x7(rectified, lr_nc, stride=1)

            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            if use_resize_conv:
                output = resizeconv(rectified, out_channels)
            else:
                output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout is not None:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            print(output.shape)
            layers.append(output)

    with tf.variable_scope("x_decoder_%d" % (len(layer_specs) + 1)):
        if lr_nc > 0 and lr_pos == 0:
            lr_output = conv7x7(rectified, lr_nc, stride=1)
        else:
            lr_output = None

    if output_num == 1:
        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("x_decoder_%d" % (len(layer_specs) + 1)):
            input = layers[-1]
            rectified = tf.nn.relu(input)
            output = conv7x7(rectified, generator_outputs_channels, stride=1)
            if activation:
                output = activation(output)
            layers.append(output)
        print(output.shape)
        return output, None
    else:
        layer_1 = layers[-1]
        outputs = []
        for i in range(output_num):
            # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
            with tf.variable_scope("x_decoder_%d" % (len(layer_specs) + 1) + str(i)):
                input = layer_1
                rectified = tf.nn.relu(input)
                output = conv7x7(rectified, generator_outputs_channels, stride=1)
                if activation:
                    output = activation(output)
                outputs.append(output)
        outputs = tuple(outputs)
        return outputs, None

def generate_revgan_y_decoder(generator_inputs, generator_outputs_channels, ngf=64, output_num=1,
                              activation=tf.tanh, use_resize_conv=False, lr_nc=0, lr_pos=0):
    print("y_decoder")
    print(generator_inputs.shape)
    layers = []
    layers.append(generator_inputs)

    layer_specs = [
        (ngf * 2, None),  # decoder_3: [batch, 128, 128, ngf * 4] => [batch, 256, 256, ngf * 2]
        (ngf, None),  # decoder_2: [batch, 256, 256, ngf * 2] => [batch, 512, 512, ngf]
    ]

    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        with tf.variable_scope("y_decoder_%d" % (decoder_layer + 1)):

            input = layers[-1]

            rectified = tf.nn.relu(input)

            if lr_nc > 0 and lr_pos == len(layer_specs) - decoder_layer:
                lr_output = conv7x7(rectified, lr_nc, stride=1)

            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            if use_resize_conv:
                output = resizeconv(rectified, out_channels)
            else:
                output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout is not None:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            print(output.shape)
            layers.append(output)

    with tf.variable_scope("x_decoder_%d" % (len(layer_specs) + 1)):
        if lr_nc > 0 and lr_pos == 0:
            lr_output = conv7x7(rectified, lr_nc, stride=1)
        else:
            lr_output = None

    if output_num == 1:
        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("y_decoder_%d" % (len(layer_specs) + 1)):
            input = layers[-1]
            rectified = tf.nn.relu(input)
            output = conv7x7(rectified, generator_outputs_channels, stride=1)
            if activation:
                output = activation(output)
            layers.append(output)
        print(output.shape)
        if(lr_output is not None):
            return output, lr_output
        else:
            return output, None
    else:
        layer_1 = layers[-1]
        outputs = []
        for i in range(output_num):
            # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
            with tf.variable_scope("y_decoder_%d" % (len(layer_specs) + 1) + str(i)):
                input = layer_1
                rectified = tf.nn.relu(input)
                output = conv7x7(rectified, generator_outputs_channels, stride=1)
                if activation:
                    output = activation(output)
                outputs.append(output)
        outputs = tuple(outputs)
        if (lr_output is not None):
            return output, lr_output
        else:
            return outputs, None

def generate_revgan_x_autoencoder(generator_inputs, generator_outputs_channels, revnet, ngf=64, dropout_prob=0.5, output_num=1,
                              activation=tf.tanh, use_resize_conv=False, lr_inputs=None, lr_pos=0):
    print("x autoencoder generator")
    layers = []
    enc_output = generate_revgan_x_encoder(generator_inputs, ngf, lr_inputs, lr_pos)
    layers.append(enc_output)

    dec_input = layers[-1]
    dec_output, lr_output = generate_revgan_x_decoder(dec_input, generator_outputs_channels, ngf, output_num, activation, use_resize_conv)

    return dec_output

def generate_revgan_y_autoencoder(generator_inputs, generator_outputs_channels, revnet, ngf=64, dropout_prob=0.5, output_num=1,
                              activation=tf.tanh, use_resize_conv=False, lr_inputs=None, lr_pos=0):
    print("y autoencoder generator")
    layers = []
    enc_output = generate_revgan_y_encoder(generator_inputs, ngf, lr_inputs, lr_pos)
    layers.append(enc_output)

    dec_input = layers[-1]
    dec_output, lr_output = generate_revgan_y_decoder(dec_input, generator_outputs_channels, ngf, output_num, activation, use_resize_conv)

    return dec_output


def generate_revgan_generator(generator_inputs, generator_outputs_channels, revnet, ngf=64, dropout_prob=0.5, output_num=1,
                              activation=tf.tanh, use_resize_conv=False, lr_inputs=None, lr_pos=0):
    print("forward generator")
    layers = []
    enc_output = generate_revgan_x_encoder(generator_inputs, ngf, lr_inputs, lr_pos)
    layers.append(enc_output)

    in_1, in_2 = tf.split(layers[-1], num_or_size_splits=2, axis=3)
    in_1 = tf.identity(in_1, name="revnet_input_1")
    in_2 = tf.identity(in_2, name="revnet_input_2")
    layers.append((in_1, in_2))

    revnet_input = (in_1, in_2)

    revnet_output = revnet.forward_pass(revnet_input)

    layers.append(revnet_output)

    out_1, out_2 = layers[-1]
    out_1 = tf.identity(out_1, name="revnet_output_1")
    out_2 = tf.identity(out_2, name="revnet_output_2")
    layers.append(tf.concat([out_1, out_2], 3))

    dec_input = layers[-1]
    dec_output, lr_output = generate_revgan_y_decoder(dec_input, generator_outputs_channels, ngf, output_num, activation, use_resize_conv)

    return dec_output

def generate_revgan_generator_backward(generator_inputs, generator_outputs_channels, revnet, ngf=64, dropout_prob=0.5, output_num=1,
                              activation=tf.tanh, use_resize_conv=False, lr_inputs=None, lr_pos=0):
    print("backward generator")
    layers = []
    enc_output = generate_revgan_y_encoder(generator_inputs, ngf, lr_inputs, lr_pos)
    layers.append(enc_output)

    in_1, in_2 = tf.split(layers[-1], num_or_size_splits=2, axis=3)
    in_1 = tf.identity(in_1, name="backward_revnet_input_1")
    in_2 = tf.identity(in_2, name="backward_revnet_input_2")
    layers.append((in_1, in_2))

    revnet_input = (in_1, in_2)

    revnet_output = revnet.backward_pass(revnet_input)

    layers.append(revnet_output)

    out_1, out_2 = layers[-1]
    out_1 = tf.identity(out_1, name="backward_revnet_output_1")
    out_2 = tf.identity(out_2, name="backward_revnet_output_2")
    layers.append(tf.concat([out_1, out_2], 3))

    dec_input = layers[-1]
    dec_output, lr_output = generate_revgan_x_decoder(dec_input, generator_outputs_channels, ngf, output_num, activation, use_resize_conv)

    return dec_output


def generate_unet(generator_inputs, generator_outputs_channels, ngf=64, bayesian_dropout=False, dropout_prob=0.5, output_num=1, activation=tf.tanh, use_resize_conv=False, lr_inputs=None, lr_pos=0):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        if lr_inputs is not None and lr_pos == 0:
            generator_inputs = tf.concat([generator_inputs, lr_inputs], axis=3)
        output = conv(generator_inputs, ngf, stride=2)
        if lr_inputs is not None and lr_pos == 1:
            layers.append(tf.concat([lr_inputs, output], axis=3))
        else:
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
            if bayesian_dropout and (len(layers) + 1) in [8, 7, 6, 5, 4]:
                output = tf.nn.dropout(output, keep_prob=1 - dropout_prob)
            if lr_inputs is not None and lr_pos == len(layers) + 1:
                layers.append( tf.concat([lr_inputs, output], axis=3))
            else:
                layers.append(output)

    layer_specs = [
        (ngf * 8, dropout_prob),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, dropout_prob),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, dropout_prob),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, None),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, None),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, None),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, None),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
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
            if use_resize_conv:
                output = resizeconv(rectified, out_channels)
            else:
                output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if bayesian_dropout and (skip_layer + 1) in [8, 7, 6, 5, 4]:
                output = tf.nn.dropout(output, keep_prob=1 - dropout_prob)
            else:
                if dropout is not None:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    if output_num == 1:
        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            if use_resize_conv:
                output = resizeconv(rectified, generator_outputs_channels)
            else:
                output = deconv(rectified, generator_outputs_channels)
            if activation:
                output = activation(output)
            layers.append(output)
        return output
    else:
        layer_1 = layers[-1]
        outputs = []
        for i in range(output_num):
            # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
            with tf.variable_scope("decoder_1_"+str(i)):
                input = tf.concat([layer_1, layers[0]], axis=3)
                rectified = tf.nn.relu(input)
                if use_resize_conv:
                    output = resizeconv(rectified, generator_outputs_channels)
                else:
                    output = deconv(rectified, generator_outputs_channels)
                if activation:
                    output = activation(output)
                outputs.append(output)
        outputs = tuple(outputs)
        return outputs


def generate_punet(generator_inputs, controls, generator_outputs_channels, ngf=64, bayesian_dropout=False, dropout_prob=0.5, output_num=1, activation=tf.tanh, use_resize_conv=False, lr_inputs=None, lr_pos=0):
    assert controls is not None
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        if lr_inputs is not None and lr_pos == 0:
            generator_inputs = tf.concat([generator_inputs, lr_inputs], axis=3)
        output = conv(generator_inputs, ngf, stride=2)
        output = output + conv1x1(controls, ngf)
        if lr_inputs is not None and lr_pos == 1:
            layers.append( tf.concat([lr_inputs, output], axis=3))
        else:
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
            convolved = convolved + conv1x1(controls, out_channels)
            output = batchnorm(convolved)
            if bayesian_dropout and (len(layers) + 1) in [8, 7, 6, 5, 4]:
                output = tf.nn.dropout(output, keep_prob=1- dropout_prob)
            if lr_inputs is not None and lr_pos == len(layers):
                layers.append( tf.concat([lr_inputs, output], axis=3))
            else:
                layers.append(output)


    layer_specs = [
        (ngf * 8, dropout_prob),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, dropout_prob),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, dropout_prob),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, None),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, None),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, None),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, None),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
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
            if use_resize_conv:
                output = resizeconv(rectified, out_channels)
            else:
                output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if bayesian_dropout and (skip_layer + 1) in [8, 7, 6, 5, 4]:
                output = tf.nn.dropout(output, keep_prob=1 - dropout_prob)
            else:
                if dropout is not None:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    if output_num == 1:
        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            if use_resize_conv:
                output = resizeconv(rectified, generator_outputs_channels)
            else:
                output = deconv(rectified, generator_outputs_channels)
            if activation:
                output = activation(output)
            layers.append(output)
        return layers[-1]
    else:
        layer_1 = layers[-1]
        outputs = []
        for i in range(output_num):
            # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
            with tf.variable_scope("decoder_1_"+str(i)):
                input = tf.concat([layer_1, layers[0]], axis=3)
                rectified = tf.nn.relu(input)
                if use_resize_conv:
                    output = resizeconv(rectified, generator_outputs_channels)
                else:
                    output = deconv(rectified, generator_outputs_channels)
                if activation:
                    output = activation(output)
                outputs.append(output)
        outputs = tuple(outputs)
        return outputs


def generate_squirrel_model(lr_inputs, outputs, targets, lr_scale=1.0, ndf=8, lr=0.0002, beta1=0.5, dropout_prob=0.4, lr_loss_mode='lr_inputs'):
    lr_pos = int(np.log2(1.0/lr_scale))
    assert lr_inputs is not None and lr_pos > 0

    with tf.name_scope("real_squirrel_discriminator"):
        with tf.variable_scope("squirrel_discriminator"):
            lr_predict_real = generate_squirrel_discriminator(targets, ndf, discriminator_layer_num=lr_pos, dropout_prob=dropout_prob)

    with tf.name_scope("fake_squirrel_discriminator"):
        with tf.variable_scope("squirrel_discriminator", reuse=True):
            lr_predict_fake = generate_squirrel_discriminator(outputs, ndf, discriminator_layer_num=lr_pos, dropout_prob=dropout_prob)

    with tf.name_scope("squirrel_discriminator_loss"):
        cond = tf.equal(tf.reduce_min(lr_inputs), tf.reduce_max(lr_inputs))
        mask = tf.cond(cond, lambda: tf.constant(0.0), lambda: tf.constant(1.0))
        squirrel_discrim_loss = mask * (1 - tf_ms_ssim(lr_predict_real, lr_inputs, mean_metric=True))
        # mask * tf_ms_ssim_l1_loss(lr_predict_real, lr_inputs, mean_metric=True, alpha=0.99)
        #1 - tf_ssim(lr_predict_real, lr_inputs, mean_metric=True, filter_size=15, filter_sigma=2)
        #tf_ms_ssim(lr_predict_real, lr_inputs, weights=[0.0448, 0.2856, 0.3001], filter_size=15, filter_sigma=2)
        ms_ssim_loss = mask * (1 - tf_ms_ssim(lr_predict_fake, lr_inputs, mean_metric=False))
        squirrel_error_map = ms_ssim_loss * (lr_predict_fake + lr_inputs)
        if lr_loss_mode == 'lr_input':
            # mask * tf_ms_ssim_l1_loss(lr_predict_fake, lr_inputs, mean_metric=False, alpha=0.99)  # tf_ms_ssim_l1_loss(lr_predict_fake, lr_inputs, mean_metric=False, alpha=0.95)
            gen_loss_squirrel = tf.reduce_mean(ms_ssim_loss)
        elif lr_loss_mode == 'lr_predict':
            ms_ssim_loss_pred = mask * (1 - tf_ms_ssim(lr_predict_fake, lr_predict_real, mean_metric=False))
            # mask * tf_ms_ssim_l1_loss(lr_predict_fake, lr_inputs, mean_metric=False, alpha=0.99)  # tf_ms_ssim_l1_loss(lr_predict_fake, lr_inputs, mean_metric=False, alpha=0.95)
            gen_loss_squirrel = tf.reduce_mean(ms_ssim_loss_pred)
        else:
            raise Exception('unsupported mode.')
        #
        #squirrel_error_map = 1 - tf_ssim(lr_inputs, lr_predict_fake, mean_metric=False, filter_size=15, filter_sigma=2)

    with tf.name_scope("squirrel_discriminator_train"):
        cond = tf.equal(tf.reduce_min(lr_inputs), tf.reduce_max(lr_inputs))
        squirrel_discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("squirrel_discriminator")]
        squirrel_discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        squirrel_discrim_grads_and_vars = squirrel_discrim_optim.compute_gradients(squirrel_discrim_loss, var_list=squirrel_discrim_tvars)
        squirrel_discrim_train = tf.cond(cond, lambda: tf.constant(True), lambda:  squirrel_discrim_optim.apply_gradients(squirrel_discrim_grads_and_vars))

    squirrel_train = ( squirrel_discrim_train, squirrel_discrim_grads_and_vars )
    squirrel_output_fetches = ( squirrel_discrim_loss, gen_loss_squirrel, squirrel_error_map, lr_predict_real, lr_predict_fake )
    return squirrel_train, squirrel_output_fetches


def generate_discriminator(discrim_inputs, discrim_targets, ndf=64, discriminator_layer_num=3, no_lsgan=False):
    n_layers = discriminator_layer_num # default value 3
    layers = []

    #fsize = 11
    #window = _tf_fspecial_gauss(fsize, 3)
    #discrim_inputs = tf.pad(discrim_inputs, [[0, 0], [fsize//2, fsize//2], [fsize//2, fsize//2], [0, 0]], mode="REFLECT")
    #discrim_inputs = tf.nn.conv2d(discrim_inputs, window, strides=[1,1,1,1], padding='VALID')

    #discrim_targets = tf.pad(discrim_targets, [[0, 0], [fsize//2, fsize//2], [fsize//2, fsize//2], [0, 0]], mode="REFLECT")
    #discrim_targets = tf.nn.conv2d(discrim_targets, window, strides=[1,1,1,1], padding='VALID')

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    if discrim_targets is not None:
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)
    else:
        input = discrim_inputs

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(input, ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

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

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        if not no_lsgan:
            output = convolved
        else:
            output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]


def generate_squirrel_discriminator(discrim_inputs, ndf=64, discriminator_layer_num=3, dropout_prob=0.4):
    n_layers = discriminator_layer_num # default value 3
    layers = []

    with tf.variable_scope("layer_1"):
        convolved = tf.layers.conv2d(discrim_inputs, ndf, 5, padding='same')
        convolved = tf.nn.dropout(convolved, keep_prob=1 - dropout_prob)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            if i>3:
                out_channels /= min(2**(i-3), 8)
            convolved = tf.layers.conv2d(layers[-1], out_channels, 5, padding='same')
            pooled = tf.layers.max_pooling2d(convolved, 2, 2)
            normalized = batchnorm(pooled)
            normalized = tf.nn.dropout(normalized, keep_prob=1 - dropout_prob)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = tf.layers.conv2d(layers[-1], int(discrim_inputs.get_shape()[3]), 5, padding='same')
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]


def one_hot_encoding(label_batch, classes = 2):
    label_batch = tf.to_int32(label_batch)
    sparse_labels = tf.reshape(label_batch, [-1, 1])
    derived_size = tf.shape(label_batch)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat([indices, sparse_labels], 1)
    outshape = tf.stack([derived_size, classes])
    labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
    labels = tf.reshape(labels, [tf.shape(label_batch)[0], 1, 1, classes])
    return labels


def create_unet_model(inputs, targets, controls, channel_masks, ngf=64, ndf=64, output_uncertainty=False, bayesian_dropout=False, use_resize_conv=False,
                      dropout_prob=0.5, lr=0.0002, beta1=0.5, lambda_tv=0, use_ssim=False, use_punet=False,
                      control_nc=0, control_classes=0, use_squirrel=False, squirrel_weight=0.5, lr_nc=0, lr_scale=1, lr_loss_mode='lr_inputs'):
    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        if use_punet:
            assert control_nc >0
            if control_classes is not None and control_classes > 0 :
                encoded = one_hot_encoding(controls[:, :, :, 0:1], classes=control_classes)
                if control_nc > 1:
                    controls = tf.concat([encoded, controls[:, :, :, 1:]], axis=3)
                else:
                    controls = encoded
        else:
            controls = None

        raw_inputs = inputs
        targets = preprocess(targets, mode='min_max[0,1]', range_lim=TARGET_RANGE_LIM)
        if lr_nc > 0:
            inputs_sr, scaled_lr_inputs = raw_inputs[:, :, :, :-lr_nc], preprocess(raw_inputs[:, :, :, -lr_nc:], mode='min_max[0,1]', range_lim=LR_RANGE_LIM)
            _b, _h, _w, _ch = scaled_lr_inputs.get_shape().as_list()
            lr_pos = int(np.log2(1.0/lr_scale))
            lr_inputs = deprocess(tf.image.resize_images(scaled_lr_inputs, size=(_h//(2**lr_pos), _w//(2**lr_pos)), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
            inputs = tf.concat([preprocess(inputs_sr, mode='mean_std'), scaled_lr_inputs], axis=3)
        else:
            scaled_lr_inputs = None
            lr_inputs = None
            inputs_sr = inputs
            inputs = preprocess(inputs, mode='mean_std')

        inputs = inputs * channel_masks

        if output_uncertainty:
            output_num = 2
            if use_punet:
                outputs, log_sigma_square = generate_punet(inputs, controls, out_channels, ngf, bayesian_dropout=bayesian_dropout, dropout_prob=dropout_prob, output_num=output_num, activation=None, use_resize_conv=use_resize_conv)
            else:
                outputs, log_sigma_square  = generate_unet(inputs, out_channels, ngf, bayesian_dropout=bayesian_dropout, dropout_prob=dropout_prob, output_num=output_num, activation=None, use_resize_conv=use_resize_conv)
            # apply activation
            outputs, log_sigma_square = tf.tanh(outputs), log_sigma_square
            sigma = tf.sqrt(tf.exp(log_sigma_square))
        else:
            output_num = 1
            if use_punet:
                outputs = generate_punet(inputs, controls, out_channels, ngf, bayesian_dropout=bayesian_dropout, dropout_prob=dropout_prob, output_num=output_num, use_resize_conv=use_resize_conv)
            else:
                outputs = generate_unet(inputs, out_channels, ngf, bayesian_dropout=bayesian_dropout, dropout_prob=dropout_prob, output_num=output_num, use_resize_conv=use_resize_conv)
            sigma = None

    if use_squirrel:
        assert lr_inputs is not None
        ndfs = ndf//8 if ndf//8>0 else 1
        _squirrel_train, _squirrel_output_fetches = generate_squirrel_model(lr_inputs, outputs, targets, lr_scale=lr_scale, ndf=ndfs, lr=lr, beta1=beta1, dropout_prob=dropout_prob, lr_loss_mode=lr_loss_mode)
        squirrel_discrim_train, squirrel_discrim_grads_and_vars = _squirrel_train
        squirrel_discrim_loss, gen_loss_squirrel, squirrel_error_map, lr_predict_real, lr_predict_fake = _squirrel_output_fetches
    else:
        squirrel_discrim_train, squirrel_discrim_grads_and_vars = (None, ) * 2
        squirrel_discrim_loss, gen_loss_squirrel, squirrel_error_map, lr_predict_real, lr_predict_fake = (None, ) * 5

    dp_outputs = deprocess(outputs)
    dp_targets = deprocess(targets)
    with tf.name_scope("generator_loss"):
        gen_loss_L2 = None
        gen_loss_L1 = None
        gen_loss_SSIM = None
        if use_ssim:
            if output_uncertainty:
                if use_ssim == 'ms_ssim':
                    ssim_map = tf_ms_ssim(dp_targets, dp_outputs, mean_metric=False)
                    ms_ssim_mean = tf.reduce_mean(ssim_map)
                    gen_loss = (1 - ms_ssim_mean) + tf.reduce_mean((1 - ssim_map)*tf.exp(-log_sigma_square)+log_sigma_square)
                elif use_ssim == 'ssim_l1':
                    ssim_l1, ssim_loss = tf_ssim_l1_loss(dp_targets, dp_outputs, mean_metric=False, filter_size=21, filter_sigma=3, alpha=0.84)
                    gen_loss = tf.reduce_mean(ssim_l1*tf.exp(-log_sigma_square)+log_sigma_square)
                    gen_loss_SSIM = tf.reduce_mean(ssim_loss)
                elif use_ssim == 'ms_ssim_l1':
                    raise NotImplemented
                elif use_ssim == 'ssim_l1_fpp':
                    raise NotImplemented
                else:
                    gen_loss = tf.reduce_mean((tf.sigmoid(log_sigma_square) - tf_ssim(dp_targets, dp_outputs, mean_metric=False, filter_size=21, filter_sigma=3))) + tf.reduce_mean(tf.square(tf.sigmoid(log_sigma_square) - 1)) *10
            else:
                if use_ssim == 'ms_ssim':
                    gen_loss = 1 - tf_ms_ssim(dp_targets, dp_outputs)
                elif use_ssim == 'ms_ssim_l1':
                    gen_loss = tf_ms_ssim_l1_loss(dp_targets, dp_outputs)
                    gen_loss_SSIM = gen_loss
                elif use_ssim == 'ssim_l1':
                    gen_loss, ssim_loss = tf_ssim_l1_loss(dp_targets, dp_outputs, mean_metric=True, filter_size=21, filter_sigma=3, alpha=0.84)
                    gen_loss_SSIM = ssim_loss
                elif use_ssim == 'ssim_l1_fpp':
                    assert controls is not None
                    ctrl = controls[:, :, :, -1:]
                    #ssim_l1 = tf_ssim_l1_loss(dp_targets, dp_outputs, mean_metric=True, filter_size=21, filter_sigma=3, alpha=0.84)
                    #gen_loss = ssim_l1 + tf.reduce_mean(tf.nn.relu((outputs-targets) /2 * ctrl)) * 10.0
                    ssim_l1, ssim_loss = tf_ssim_l1_loss(dp_targets, dp_outputs, mean_metric=False, filter_size=21, filter_sigma=3, alpha=0.84)
                    gen_loss = tf.reduce_mean(ssim_l1) + tf.reduce_mean(dp_outputs * ssim_l1 * ctrl) * 10
                    gen_loss_SSIM = tf.reduce_mean(ssim_loss)
                elif use_ssim == 'io_ssim':
                    gen_loss_SSIM = tf.reduce_mean(1 - tf_ssim(dp_targets, dp_outputs, mean_metric=False, filter_size=21, filter_sigma=3))
                    if use_punet:
                        ctrl = controls[:, :, :, -1:]
                        si = tf.reduce_mean((1 - tf_ssim(inputs_sr, dp_outputs, mean_metric=False, filter_size=21, filter_sigma=3)) * ctrl)
                        gen_loss = gen_loss_SSIM + si * 10.0
                    else:
                        si = tf.reduce_mean(1 - tf_ssim(inputs_sr, dp_outputs, mean_metric=False, filter_size=21, filter_sigma=3))
                        gen_loss = gen_loss_SSIM + si * 0.5
                else:
                    gen_loss = tf.reduce_mean(1 - tf_ssim(dp_targets, dp_outputs, mean_metric=False, filter_size=21, filter_sigma=3))
        else:
            if output_uncertainty:
                gen_loss = tf.reduce_mean(tf.square(targets - outputs)*tf.exp(-log_sigma_square)+log_sigma_square)
            else:
                gen_loss_L2 = tf.reduce_mean(tf.square(targets - outputs))
                gen_loss = gen_loss_L2

        if lambda_tv > 0:
            # lambda_tv = 1e-3
            loss_tv = lambda_tv *  tf.reduce_mean(tf.image.total_variation(outputs))
            gen_loss = gen_loss + loss_tv

        total_loss = gen_loss

        if use_squirrel and squirrel_weight > 0:
            total_loss = total_loss + gen_loss_squirrel * squirrel_weight

        if gen_loss_SSIM is None:
            gen_loss_SSIM = 1 - tf_ssim(dp_targets, dp_outputs, mean_metric=True, filter_size=21, filter_sigma=3)

        if gen_loss_L2 is None:
            gen_loss_L2 = tf.reduce_mean(tf.square(targets - outputs))

    with tf.name_scope("generator_train"):
        if squirrel_discrim_train is not None:
            with tf.control_dependencies([squirrel_discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(lr, beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(total_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
        else:
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(total_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    _losses = [gen_loss, gen_loss_L2, gen_loss_SSIM]
    if gen_loss_squirrel is not None:
        _losses.append(gen_loss_squirrel)
    if squirrel_discrim_loss is not None:
        _losses.append(squirrel_discrim_loss)
    update_losses = ema.apply(list(set(_losses)))

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        type='UNET',
        predict_real=None,
        predict_fake=None,
        inputs=inputs_sr,
        lr_inputs=lr_inputs,
        lr_predict_real=lr_predict_real,
        lr_predict_fake=lr_predict_fake,
        squirrel_error_map=squirrel_error_map,
        squirrel_discrim_loss=ema.average(squirrel_discrim_loss) if squirrel_discrim_loss is not None else None,
        squirrel_discrim_grads_and_vars=squirrel_discrim_grads_and_vars,
        discrim_loss=None,
        discrim_grads_and_vars=None,
        gen_loss_GAN=None,
        gen_loss=ema.average(gen_loss),
        gen_loss_L1=gen_loss_L1,
        gen_loss_L2=ema.average(gen_loss_L2),
        gen_loss_SSIM=ema.average(gen_loss_SSIM),
        gen_loss_squirrel=ema.average(gen_loss_squirrel) if gen_loss_squirrel is not None else None,
        gen_grads_and_vars=gen_grads_and_vars,
        losses = {'gen_loss': gen_loss, 'discrim_loss': discrim_loss,
                  'gen_loss_L1': gen_loss_L1, 'gen_loss_L2': gen_loss_L2,
                  'gen_loss_SSIM': gen_loss_SSIM, 'gen_loss_squirrel': gen_loss_squirrel,
                  'squirrel_discrim_loss': squirrel_discrim_loss},
        outputs=dp_outputs,
        targets=dp_targets,
        uncertainty=sigma,
        squirrel_discrim_train=squirrel_discrim_train,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def create_pix2pix_model(inputs, targets, controls, channel_masks, ngf=64, ndf=64, discriminator_layer_num=3, output_uncertainty=False,
                         bayesian_dropout=False, use_resize_conv=False, no_lsgan=False, dropout_prob=0.5,
                         gan_weight=1.0, l1_weight=40.0, lr=0.0002, beta1=0.5, lambda_tv=0, use_ssim=False,
                         use_punet=False, control_nc=0, control_classes=0, use_gaussd=False, lr_nc=0, lr_scale=1,
                         use_squirrel=False, squirrel_weight=20.0, lr_loss_mode='lr_inputs'):
    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        if use_punet:
            assert control_nc >0
            if control_classes is not None and control_classes > 0 :
                encoded = one_hot_encoding(controls[:, :, :, 0:1], classes=control_classes)
                if control_nc > 1:
                    controls = tf.concat([encoded, controls[:, :, :, 1:]], axis=3)
                else:
                    controls = encoded
        else:
            controls = None

        raw_inputs = inputs
        targets = preprocess(targets, mode='min_max[0,1]', range_lim=TARGET_RANGE_LIM)
        if lr_nc > 0:
            inputs_sr, scaled_lr_inputs = raw_inputs[:, :, :, :-lr_nc], preprocess(raw_inputs[:, :, :, -lr_nc:], mode='min_max[0,1]', range_lim=LR_RANGE_LIM)
            _b, _h, _w, _ch = scaled_lr_inputs.get_shape().as_list()
            lr_pos = int(np.log2(1.0/lr_scale))
            lr_inputs = deprocess(tf.image.resize_images(scaled_lr_inputs, size=(_h//(2**lr_pos), _w//(2**lr_pos)), method=tf.image.ResizeMethod.BILINEAR))
            inputs = tf.concat([preprocess(inputs_sr, mode='mean_std'), scaled_lr_inputs], axis=3)
        else:
            scaled_lr_inputs = None
            lr_inputs = None
            inputs_sr = inputs
            inputs = preprocess(inputs, mode='mean_std')
        inputs = inputs * channel_masks

        if output_uncertainty:
            output_num = 2
            if use_punet:
                outputs, log_sigma_square  = generate_punet(inputs, controls, out_channels, ngf, bayesian_dropout=bayesian_dropout, dropout_prob=dropout_prob, output_num=output_num, activation=None, use_resize_conv=use_resize_conv)
            else:
                outputs, log_sigma_square  = generate_unet(inputs, out_channels, ngf, bayesian_dropout=bayesian_dropout, dropout_prob=dropout_prob, output_num=output_num, activation=None, use_resize_conv=use_resize_conv)
            # apply activation
            outputs, log_sigma_square = tf.tanh(outputs), log_sigma_square
            sigma = tf.sqrt(tf.exp(log_sigma_square))
        else:
            output_num = 1
            if use_punet:
                outputs = generate_punet(inputs, controls, out_channels, ngf, bayesian_dropout=bayesian_dropout, dropout_prob=dropout_prob, output_num=output_num, use_resize_conv=use_resize_conv)
            else:
                outputs = generate_unet(inputs, out_channels, ngf, bayesian_dropout=bayesian_dropout, dropout_prob=dropout_prob, output_num=output_num, use_resize_conv=use_resize_conv)
            sigma = None


    with tf.name_scope("discriminator_inputs"):
        _, h, w, _ = inputs.get_shape().as_list()
        if use_punet:
            control_channels = tf.tile(controls, [1, h, w, 1])
            inputs = tf.concat([inputs, control_channels], axis=3)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            if use_gaussd:
                predict_real = generate_discriminator(tf_gauss_conv(inputs), tf_gauss_conv(targets), ndf, discriminator_layer_num=discriminator_layer_num, no_lsgan=no_lsgan)
            else:
                predict_real = generate_discriminator(inputs, targets, ndf, discriminator_layer_num=discriminator_layer_num, no_lsgan=no_lsgan)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            if use_gaussd:
                predict_fake = generate_discriminator(tf_gauss_conv(inputs), tf_gauss_conv(outputs), ndf, discriminator_layer_num=discriminator_layer_num, no_lsgan=no_lsgan)
            else:
                predict_fake = generate_discriminator(inputs, outputs, ndf, discriminator_layer_num=discriminator_layer_num, no_lsgan=no_lsgan)


    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        if not no_lsgan:
            discrim_loss = tf.reduce_mean(tf.square(predict_real - 1)) + tf.reduce_mean(tf.square(predict_fake))
        else:
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    if use_squirrel:
        ndfs = ndf//8 if ndf//8>0 else 1
        _squirrel_train, _squirrel_output_fetches = generate_squirrel_model(lr_inputs, outputs, targets, lr_scale=lr_scale, ndf=ndfs, lr=lr, beta1=beta1, dropout_prob=dropout_prob, lr_loss_mode=lr_loss_mode)
        squirrel_discrim_train, squirrel_discrim_grads_and_vars = _squirrel_train
        squirrel_discrim_loss, gen_loss_squirrel, squirrel_error_map, lr_predict_real, lr_predict_fake = _squirrel_output_fetches
    else:
        squirrel_discrim_train, squirrel_discrim_grads_and_vars = (None, ) * 2
        squirrel_discrim_loss, gen_loss_squirrel, squirrel_error_map, lr_predict_real, lr_predict_fake = (None, ) * 5

    dp_outputs = deprocess(outputs)
    dp_targets = deprocess(targets)
    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_L2 = None
        gen_loss_L1 = None
        gen_loss_SSIM = None
        if not no_lsgan:
            if output_uncertainty:
                # "VALID" mode in convolution only ever drops the right-most columns (or bottom-most rows).
                scaled_log_sigma_square = tf.nn.avg_pool(log_sigma_square, (1, 2**3, 2**3, 1), [1, 2**3, 2**3, 1], padding="VALID")[:, :-2, :-2, :]
                gen_loss_GAN = tf.reduce_mean(tf.square(predict_fake-1) * tf.exp(-scaled_log_sigma_square) + scaled_log_sigma_square)
            else:
                gen_loss_GAN = tf.reduce_mean(tf.square(predict_fake-1))
        else:
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        if use_ssim:
            if output_uncertainty:
                if use_ssim == 'ms_ssim':
                    ms_ssim_map = tf_ms_ssim(dp_targets, dp_outputs)
                    gen_loss_SSIM = 1 - tf.reduce_mean(ms_ssim_map)
                    gen_loss = gen_loss_SSIM + tf.reduce_mean((1 - ms_ssim_map) * tf.exp(-log_sigma_square)+log_sigma_square)
                elif use_ssim == 'ssim_l1':
                    ssim_l1, ssim_loss = tf_ssim_l1_loss(dp_targets, dp_outputs, mean_metric=False, filter_size=21, filter_sigma=3, alpha=0.84)
                    gen_loss = tf.reduce_mean(ssim_l1 * tf.exp(-log_sigma_square) + log_sigma_square)
                    gen_loss_SSIM = tf.reduce_mean(ssim_loss)
                elif use_ssim == 'ms_ssim_l1':
                    raise NotImplemented
                elif use_ssim == 'io_ssim':
                    raise NotImplemented
                elif use_ssim == 'ssim_l1_fpp':
                    raise NotImplemented
                else:
                    gen_loss = tf.reduce_mean((1 - tf_ssim(dp_targets, dp_outputs, mean_metric=False, filter_size=21, filter_sigma=3))*tf.exp(-log_sigma_square)+log_sigma_square)
            else:
                if use_ssim == 'ms_ssim':
                    gen_loss_SSIM = 1 - tf_ms_ssim(dp_targets, dp_outputs)
                    gen_loss = gen_loss_SSIM
                elif use_ssim == 'ms_ssim_l1':
                    gen_loss = tf_ms_ssim_l1_loss(dp_targets, dp_outputs)
                    gen_loss_SSIM = gen_loss
                elif use_ssim == 'ssim_l1':
                    gen_loss, ssim_loss = tf_ssim_l1_loss(dp_targets, dp_outputs, mean_metric=True, filter_size=21, filter_sigma=3, alpha=0.84)
                    gen_loss_SSIM = ssim_loss
                elif use_ssim == 'io_ssim':
                    raise NotImplemented
                elif use_ssim == 'ssim_l1_fpp':
                    assert controls is not None
                    ctrl = controls[:, :, :, -1:]
                    # ssim_l1, ssim_loss = tf_ssim_l1_loss(targets, outputs, mean_metric=True, filter_size=21, filter_sigma=3, alpha=0.84)
                    # gen_loss_L2 = ssim_l1 + tf.reduce_mean(tf.nn.relu((outputs-targets) /2 * ctrl)) * 10.0
                    ssim_l1, ssim_loss = tf_ssim_l1_loss(dp_targets, dp_outputs, mean_metric=False, filter_size=21, filter_sigma=3, alpha=0.84)
                    gen_loss = tf.reduce_mean(ssim_l1) + tf.reduce_mean(dp_outputs * ssim_l1 * ctrl) * 2
                    gen_loss_SSIM = tf.reduce_mean(ssim_loss)
                else:
                    gen_loss = tf.reduce_mean(1 - tf_ssim(dp_targets, dp_outputs, mean_metric=False, filter_size=21, filter_sigma=3))
        else:
            if output_uncertainty:
                gen_loss = tf.reduce_mean(tf.abs(targets - outputs) * tf.exp(-log_sigma_square) + log_sigma_square)
            else:
                gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
                gen_loss = gen_loss_L1

        if lambda_tv > 0:
            loss_tv = lambda_tv * tf.reduce_mean(tf.image.total_variation(outputs))
            gen_loss = gen_loss + loss_tv

        total_loss = gen_loss_GAN * gan_weight + gen_loss * l1_weight

        if use_squirrel and squirrel_weight > 0:
            total_loss = total_loss + gen_loss_squirrel * squirrel_weight

        if gen_loss_SSIM is None:
            gen_loss_SSIM = 1 - tf_ssim(dp_targets, dp_outputs, mean_metric=True, filter_size=21, filter_sigma=3)
        if gen_loss_L2 is None:
            gen_loss_L2 = tf.reduce_mean(tf.square(targets - outputs))

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        dependencies = [discrim_train]
        if squirrel_discrim_train is not None:
            dependencies.append(squirrel_discrim_train)
        with tf.control_dependencies(dependencies):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(total_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    _losses = [discrim_loss, gen_loss_GAN, gen_loss, gen_loss_SSIM, gen_loss_L2]
    if gen_loss_squirrel is not None:
        _losses.append(gen_loss_squirrel)
    if squirrel_discrim_loss is not None:
        _losses.append(squirrel_discrim_loss)
    update_losses = ema.apply(list(set(_losses)))

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        type='GAN',
        predict_real=predict_real,
        predict_fake=predict_fake,
        inputs=inputs_sr,
        lr_inputs=lr_inputs,
        lr_predict_real=lr_predict_real,
        lr_predict_fake=lr_predict_fake,
        squirrel_error_map=squirrel_error_map,
        squirrel_discrim_loss=ema.average(squirrel_discrim_loss) if squirrel_discrim_loss is not None else None,
        squirrel_discrim_grads_and_vars=squirrel_discrim_grads_and_vars,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss=ema.average(gen_loss),
        gen_loss_L1=gen_loss_L1,
        gen_loss_L2=ema.average(gen_loss_L2),
        gen_loss_SSIM=ema.average(gen_loss_SSIM),
        gen_loss_squirrel=ema.average(gen_loss_squirrel) if gen_loss_squirrel is not None else None,
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=dp_outputs,
        targets=dp_targets,
        losses = {'gen_loss': gen_loss, 'discrim_loss': discrim_loss,
                  'gen_loss_L1': gen_loss_L1, 'gen_loss_L2': gen_loss_L2,
                  'gen_loss_SSIM': gen_loss_SSIM, 'gen_loss_squirrel': gen_loss_squirrel,
                  'squirrel_discrim_loss': squirrel_discrim_loss, 'gen_loss_GAN': gen_loss_GAN},
        uncertainty=sigma,
        squirrel_discrim_train=squirrel_discrim_train,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )

def create_revgan_model(inputs, targets, controls, channel_masks, ngf=64, ndf=64, discriminator_layer_num=3,
                        output_uncertainty=False,
                        bayesian_dropout=False, use_resize_conv=False, no_lsgan=False, dropout_prob=0.5,
                        gan_weight=1.0, l1_weight=40.0, lr=0.0002, beta1=0.5, lambda_tv=0, use_ssim=False,
                        use_punet=False, control_nc=0, control_classes=0, use_gaussd=False, lr_nc=0, lr_scale=1,
                        use_squirrel=False, squirrel_weight=20.0, lr_loss_mode='lr_inputs', rev_layer_num=10):
    with tf.name_scope("generator"):
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as scope:
            out_channels = int(targets.get_shape()[-1])
            revnet = ReversibleNet(rev_layer_num, 128)
            if use_punet:
                assert control_nc > 0
                if control_classes is not None and control_classes > 0:
                    encoded = one_hot_encoding(controls[:, :, :, 0:1], classes=control_classes)
                    if control_nc > 1:
                        controls = tf.concat([encoded, controls[:, :, :, 1:]], axis=3)
                    else:
                        controls = encoded
            else:
                controls = None

            raw_inputs = inputs
            targets = preprocess(targets, mode='min_max[0,1]', range_lim=TARGET_RANGE_LIM)
            if lr_nc > 0:
                inputs_sr, scaled_lr_inputs = raw_inputs[:, :, :, :-lr_nc], preprocess(raw_inputs[:, :, :, -lr_nc:],
                                                                                       mode='min_max[0,1]',
                                                                                       range_lim=LR_RANGE_LIM)
                _b, _h, _w, _ch = scaled_lr_inputs.get_shape().as_list()
                lr_pos = int(np.log2(1.0 / lr_scale))
                lr_inputs = deprocess(
                    tf.image.resize_images(scaled_lr_inputs, size=(_h // (2 ** lr_pos), _w // (2 ** lr_pos)),
                                           method=tf.image.ResizeMethod.BILINEAR))
                inputs = tf.concat([preprocess(inputs_sr, mode='mean_std'), scaled_lr_inputs], axis=3)
            else:
                scaled_lr_inputs = None
                lr_inputs = None
                inputs_sr = inputs
                inputs = preprocess(inputs, mode='mean_std')
            inputs = inputs * channel_masks

            # generate generator output
            # outputs = generate_generator(inputs, out_channels, ngf, bayesian_dropout=bayesian_dropout, dropout_prob=dropout_prob, output_num=output_num, use_resize_conv=use_resize_conv)
            if output_uncertainty:
                output_num = 2

                outputs, log_sigma_square = generate_revgan_generator(inputs, out_channels, revnet, ngf,
                                                                      dropout_prob=dropout_prob, output_num=output_num,
                                                                      activation=None, use_resize_conv=use_resize_conv)
                # apply activation
                outputs, log_sigma_square = tf.tanh(outputs), log_sigma_square
                sigma = tf.sqrt(tf.exp(log_sigma_square))
            else:
                output_num = 1

                outputs = generate_revgan_generator(inputs, out_channels, revnet, ngf,
                                                    dropout_prob=dropout_prob, output_num=output_num,
                                                    use_resize_conv=use_resize_conv)
                sigma = None

            backward_outputs = generate_revgan_generator_backward(targets, inputs.shape[-1], revnet, ngf,
                                                                      dropout_prob=dropout_prob, output_num=output_num,
                                                                       use_resize_conv=use_resize_conv)

            x_auto_outputs = generate_revgan_x_autoencoder(inputs, inputs.shape[-1], revnet, ngf,
                                                           dropout_prob=dropout_prob, output_num=output_num,
                                                            use_resize_conv=use_resize_conv)

            y_auto_outputs = generate_revgan_y_autoencoder(targets, out_channels, revnet, ngf,
                                                           dropout_prob=dropout_prob, output_num=output_num,
                                                            use_resize_conv=use_resize_conv)
            backward_outputs_lr = backward_outputs[:, :, :, -lr_nc:]
            x_auto_outputs_lr = x_auto_outputs[:, :, :, -lr_nc:]

    # with tf.name_scope("generator_reverse"):
    #    with tf.variable_scope("generator", reuse=True) as scope:
    #        outputs = generate_revgan_generator(inputs, out_channels, ngf, dropout_prob=dropout_prob, output_num=output_num, use_resize_conv=use_resize_conv)

    with tf.name_scope("discriminator_inputs"):
        _, h, w, _ = inputs.get_shape().as_list()
        if use_punet:
            control_channels = tf.tile(controls, [1, h, w, 1])
            inputs = tf.concat([inputs, control_channels], axis=3)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            if use_gaussd:
                predict_real = generate_discriminator(tf_gauss_conv(inputs), tf_gauss_conv(targets), ndf,
                                                      discriminator_layer_num=discriminator_layer_num,
                                                      no_lsgan=no_lsgan)
            else:
                predict_real = generate_discriminator(inputs, targets, ndf,
                                                      discriminator_layer_num=discriminator_layer_num,
                                                      no_lsgan=no_lsgan)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            if use_gaussd:
                predict_fake = generate_discriminator(tf_gauss_conv(inputs), tf_gauss_conv(outputs), ndf,
                                                      discriminator_layer_num=discriminator_layer_num,
                                                      no_lsgan=no_lsgan)
            else:
                predict_fake = generate_discriminator(inputs, outputs, ndf,
                                                      discriminator_layer_num=discriminator_layer_num,
                                                      no_lsgan=no_lsgan)

    with tf.name_scope("real_lr_discriminator"):
        with tf.variable_scope("lr_discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            if use_gaussd:
                lr_predict_real = generate_discriminator(tf_gauss_conv(outputs), tf_gauss_conv(scaled_lr_inputs), ndf,
                                                      discriminator_layer_num=discriminator_layer_num,
                                                      no_lsgan=no_lsgan)
            else:
                lr_predict_real = generate_discriminator(outputs, scaled_lr_inputs, ndf,
                                                      discriminator_layer_num=discriminator_layer_num,
                                                      no_lsgan=no_lsgan)

    with tf.name_scope("fake_lr_discriminator"):
        with tf.variable_scope("lr_discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            if use_gaussd:
                lr_predict_fake = generate_discriminator(tf_gauss_conv(outputs), tf_gauss_conv(backward_outputs_lr), ndf,
                                                      discriminator_layer_num=discriminator_layer_num,
                                                      no_lsgan=no_lsgan)
            else:
                lr_predict_fake = generate_discriminator(outputs, backward_outputs_lr, ndf,
                                                      discriminator_layer_num=discriminator_layer_num,
                                                      no_lsgan=no_lsgan)


    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        if not no_lsgan:
            discrim_loss = tf.reduce_mean(tf.square(predict_real - 1)) + tf.reduce_mean(tf.square(predict_fake))
        else:
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("lr_discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        if not no_lsgan:
            lr_discrim_loss = tf.reduce_mean(tf.square(lr_predict_real - 1)) + tf.reduce_mean(tf.square(lr_predict_fake))
        else:
            lr_discrim_loss = tf.reduce_mean(-(tf.log(lr_predict_real + EPS) + tf.log(1 - lr_predict_fake + EPS)))

    """with tf.name_scope("generator_x_cycle"):
        # generate x_cycle ouput ????
    with tf.name_scope("generator_y_cycle"):
        # generate y_cycle output ????"""

    # with tf.name_scope("generator_reverse"):
    # generate reverse output

    """if use_squirrel:
        ndfs = ndf//8 if ndf//8>0 else 1
        _squirrel_train, _squirrel_output_fetches = generate_squirrel_model(lr_inputs, outputs, targets, lr_scale=lr_scale, ndf=ndfs, lr=lr, beta1=beta1, dropout_prob=dropout_prob, lr_loss_mode=lr_loss_mode)
        squirrel_discrim_train, squirrel_discrim_grads_and_vars = _squirrel_train
        squirrel_discrim_loss, gen_loss_squirrel, squirrel_error_map, lr_predict_real, lr_predict_fake = _squirrel_output_fetches
    else:
        squirrel_discrim_train, squirrel_discrim_grads_and_vars = (None, ) * 2
        squirrel_discrim_loss, gen_loss_squirrel, squirrel_error_map, lr_predict_real, lr_predict_fake = (None, ) * 5"""

    dp_outputs = deprocess(outputs)
    dp_targets = deprocess(targets)
    dp_xaoutputs = deprocess(x_auto_outputs_lr)
    dp_yaoutputs = deprocess(y_auto_outputs)
    dp_backward_outputs = deprocess(backward_outputs_lr)
    dp_lr_input = deprocess(scaled_lr_inputs)
    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_L2 = None
        gen_loss_L1 = None
        gen_loss_SSIM = None
        xae_loss_total = None
        yae_loss_total = None
        bw_gen_loss_total = None
        if not no_lsgan:
            if output_uncertainty:
                # "VALID" mode in convolution only ever drops the right-most columns (or bottom-most rows).
                scaled_log_sigma_square = tf.nn.avg_pool(log_sigma_square, (1, 2 ** 3, 2 ** 3, 1),
                                                         [1, 2 ** 3, 2 ** 3, 1], padding="VALID")[:, :-2, :-2, :]
                gen_loss_GAN = tf.reduce_mean(
                    tf.square(predict_fake - 1) * tf.exp(-scaled_log_sigma_square) + scaled_log_sigma_square)
            else:
                gen_loss_GAN = tf.reduce_mean(tf.square(predict_fake - 1))
        else:
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        if use_ssim:
            if output_uncertainty:
                if use_ssim == 'ms_ssim':
                    ms_ssim_map = tf_ms_ssim(dp_targets, dp_outputs)
                    gen_loss_SSIM = 1 - tf.reduce_mean(ms_ssim_map)
                    gen_loss = gen_loss_SSIM + tf.reduce_mean(
                        (1 - ms_ssim_map) * tf.exp(-log_sigma_square) + log_sigma_square)
                elif use_ssim == 'ssim_l1':
                    ssim_l1, ssim_loss = tf_ssim_l1_loss(dp_targets, dp_outputs, mean_metric=False, filter_size=21,
                                                         filter_sigma=3, alpha=0.84)
                    gen_loss = tf.reduce_mean(ssim_l1 * tf.exp(-log_sigma_square) + log_sigma_square)
                    gen_loss_SSIM = tf.reduce_mean(ssim_loss)
                elif use_ssim == 'ms_ssim_l1':
                    raise NotImplemented
                elif use_ssim == 'io_ssim':
                    raise NotImplemented
                elif use_ssim == 'ssim_l1_fpp':
                    raise NotImplemented
                else:
                    gen_loss = tf.reduce_mean((1 - tf_ssim(dp_targets, dp_outputs, mean_metric=False, filter_size=21,
                                                           filter_sigma=3)) * tf.exp(
                        -log_sigma_square) + log_sigma_square)
            else:
                if use_ssim == 'ms_ssim':
                    gen_loss_SSIM = 1 - tf_ms_ssim(dp_targets, dp_outputs)
                    gen_loss = gen_loss_SSIM
                elif use_ssim == 'ms_ssim_l1':
                    gen_loss = tf_ms_ssim_l1_loss(dp_targets, dp_outputs)
                    gen_loss_SSIM = gen_loss
                elif use_ssim == 'ssim_l1':
                    gen_loss, ssim_loss = tf_ssim_l1_loss(dp_targets, dp_outputs, mean_metric=True, filter_size=21,
                                                          filter_sigma=3, alpha=0.84)
                    gen_loss_SSIM = ssim_loss
                elif use_ssim == 'io_ssim':
                    raise NotImplemented
                elif use_ssim == 'ssim_l1_fpp':
                    assert controls is not None
                    ctrl = controls[:, :, :, -1:]
                    # ssim_l1, ssim_loss = tf_ssim_l1_loss(targets, outputs, mean_metric=True, filter_size=21, filter_sigma=3, alpha=0.84)
                    # gen_loss_L2 = ssim_l1 + tf.reduce_mean(tf.nn.relu((outputs-targets) /2 * ctrl)) * 10.0
                    ssim_l1, ssim_loss = tf_ssim_l1_loss(dp_targets, dp_outputs, mean_metric=False, filter_size=21,
                                                         filter_sigma=3, alpha=0.84)
                    gen_loss = tf.reduce_mean(ssim_l1) + tf.reduce_mean(dp_outputs * ssim_l1 * ctrl) * 2
                    gen_loss_SSIM = tf.reduce_mean(ssim_loss)
                else:
                    gen_loss = tf.reduce_mean(
                        1 - tf_ssim(dp_targets, dp_outputs, mean_metric=False, filter_size=21, filter_sigma=3))

        else:
            if output_uncertainty:
                gen_loss = tf.reduce_mean(tf.abs(targets - outputs) * tf.exp(-log_sigma_square) + log_sigma_square)
            else:
                gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
                gen_loss = gen_loss_L1

        if not no_lsgan:
            if output_uncertainty:
                # "VALID" mode in convolution only ever drops the right-most columns (or bottom-most rows).
                gen_loss_lr_GAN = None
            else:
                gen_loss_lr_GAN = tf.reduce_mean(tf.square(lr_predict_fake - 1))
        else:
            gen_loss_lr_GAN = tf.reduce_mean(-tf.log(lr_predict_fake + EPS))

        xae_loss_SSIM = tf_ms_ssim_l1_loss(dp_xaoutputs, dp_lr_input)
        yae_loss_SSIM = tf_ms_ssim_l1_loss(dp_yaoutputs, dp_targets)
        bw_gen_loss_SSIM = tf_ms_ssim_l1_loss(dp_backward_outputs, dp_lr_input)

        xae_loss_total = xae_loss_SSIM * l1_weight
        yae_loss_total = yae_loss_SSIM * l1_weight
        bw_gen_loss_total = gen_loss_lr_GAN * gan_weight + bw_gen_loss_SSIM * l1_weight

        if lambda_tv > 0:
            loss_tv = lambda_tv * tf.reduce_mean(tf.image.total_variation(outputs))
            gen_loss = gen_loss + loss_tv

        total_loss = gen_loss_GAN * gan_weight + gen_loss * l1_weight

        if gen_loss_SSIM is None:
            gen_loss_SSIM = 1 - tf_ssim(dp_targets, dp_outputs, mean_metric=True, filter_size=21, filter_sigma=3)
        if gen_loss_L2 is None:
            gen_loss_L2 = tf.reduce_mean(tf.square(targets - outputs))

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("lr_discriminator_train"):
        lr_discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("lr_discriminator")]
        lr_discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        lr_discrim_grads_and_vars = lr_discrim_optim.compute_gradients(lr_discrim_loss, var_list=lr_discrim_tvars)
        lr_discrim_train = discrim_optim.apply_gradients(lr_discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        dependencies = [discrim_train, lr_discrim_train]
        with tf.control_dependencies(dependencies):
            # compute gradients for decoder part
            dec_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator/y_decoder")]
            rev_out_1_var = tf.get_default_graph().get_tensor_by_name("generator/generator/revnet_output_1:0")
            rev_out_2_var = tf.get_default_graph().get_tensor_by_name("generator/generator/revnet_output_2:0")
            dec_grads = tf.gradients(total_loss, [rev_out_1_var, rev_out_2_var] + dec_vars,
                                     stop_gradients=[rev_out_1_var, rev_out_2_var])
            rev_out_1_grad = dec_grads[0]
            rev_out_2_grad = dec_grads[1]
            dec_grads = dec_grads[2:]
            dec_grads_and_vars = np.array(list(zip(dec_grads, dec_vars)))

            # manual gradients for revnet
            (dy1, dy2), rev_grads_and_vars = revnet.compute_revnet_gradients_of_forward_pass(rev_out_1_var,
                                                                                             rev_out_2_var,
                                                                                             rev_out_1_grad,
                                                                                             rev_out_2_grad)
            rev_grads_and_vars = np.array(rev_grads_and_vars)

            # gradients for encoder part
            enc_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator/x_encoder")]
            rev_in_1_var = tf.get_default_graph().get_tensor_by_name("generator/generator/revnet_input_1:0")
            rev_in_2_var = tf.get_default_graph().get_tensor_by_name("generator/generator/revnet_input_2:0")
            enc_grads = tf.gradients([rev_in_1_var, rev_in_2_var], enc_vars, [dy1, dy2])
            enc_grads_and_vars = np.array(list(zip(enc_grads, enc_vars)))

            # combine all gradients in one list
            gen_grads_and_vars = np.concatenate((dec_grads_and_vars, rev_grads_and_vars, enc_grads_and_vars), axis=0)
            gen_grads_and_vars = gen_grads_and_vars.tolist()

            # apply gradients to graph
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                gen_optim = tf.train.AdamOptimizer(lr, beta1)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)


    with tf.name_scope("backward_generator_train"):
        dependencies = [discrim_train, gen_train]
        with tf.control_dependencies(dependencies):
            # compute gradients for decoder part
            dec_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator/x_decoder")]
            rev_out_1_var = tf.get_default_graph().get_tensor_by_name("generator/generator/backward_revnet_output_1:0")
            rev_out_2_var = tf.get_default_graph().get_tensor_by_name("generator/generator/backward_revnet_output_2:0")
            dec_grads = tf.gradients(bw_gen_loss_total, [rev_out_1_var, rev_out_2_var] + dec_vars, stop_gradients=[rev_out_1_var, rev_out_2_var])
            rev_out_1_grad = dec_grads[0]
            rev_out_2_grad = dec_grads[1]
            dec_grads = dec_grads[2:]
            dec_grads_and_vars = np.array(list(zip(dec_grads, dec_vars)))

            # manual gradients for revnet
            (dy1, dy2), rev_grads_and_vars = revnet.compute_revnet_gradients_of_forward_pass(rev_out_1_var,
                                                                                             rev_out_2_var,
                                                                                             rev_out_1_grad,
                                                                                             rev_out_2_grad)
            rev_grads_and_vars = np.array(rev_grads_and_vars)

            # gradients for encoder part
            enc_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator/y_encoder")]
            rev_in_1_var = tf.get_default_graph().get_tensor_by_name("generator/generator/backward_revnet_input_1:0")
            rev_in_2_var = tf.get_default_graph().get_tensor_by_name("generator/generator/backward_revnet_input_2:0")
            enc_grads = tf.gradients([rev_in_1_var, rev_in_2_var], enc_vars, [dy1, dy2])
            enc_grads_and_vars = np.array(list(zip(enc_grads, enc_vars)))

            # combine all gradients in one list
            bw_gen_grads_and_vars = np.concatenate((dec_grads_and_vars, enc_grads_and_vars), axis=0)
            bw_gen_grads_and_vars = bw_gen_grads_and_vars.tolist()

            # apply gradients to graph
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                bw_gen_optim = tf.train.AdamOptimizer(lr, beta1)
                bw_gen_train = bw_gen_optim.apply_gradients(bw_gen_grads_and_vars)

    with tf.name_scope("x_autoencoder_train"):
        dependencies = [discrim_train, gen_train, bw_gen_train]
        with tf.control_dependencies(dependencies):
            xenc_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator/x_encoder")]
            xdec_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator/x_decoder")]
            xae_vars = xenc_vars + xdec_vars
            xae_grads = tf.gradients(xae_loss_total, xae_vars)
            xae_grads_and_vars = np.array(list(zip(xae_grads, xae_vars)))
            # apply gradients to graph
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                xae_optim = tf.train.AdamOptimizer(lr, beta1)
                xae_train = xae_optim.apply_gradients(xae_grads_and_vars)

    with tf.name_scope("y_autoencoder_train"):
        dependencies = [discrim_train, gen_train, bw_gen_train, xae_train]
        with tf.control_dependencies(dependencies):
            yenc_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator/y_encoder")]
            ydec_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator/y_decoder")]
            yae_vars = yenc_vars + ydec_vars
            yae_grads = tf.gradients(yae_loss_total, yae_vars)
            yae_grads_and_vars = np.array(list(zip(yae_grads, yae_vars)))
            # apply gradients to graph
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                yae_optim = tf.train.AdamOptimizer(lr, beta1)
                yae_train = yae_optim.apply_gradients(yae_grads_and_vars)


    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        _losses = [discrim_loss, gen_loss_GAN, gen_loss, gen_loss_SSIM, gen_loss_L2, bw_gen_loss_total, xae_loss_total, yae_loss_total]
        update_losses = ema.apply(list(set(_losses)))

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        type='GAN',
        predict_real=predict_real,
        predict_fake=predict_fake,
        inputs=inputs_sr,
        lr_inputs=scaled_lr_inputs,
        lr_predict_real=lr_predict_real,
        lr_predict_fake=lr_predict_fake,
        squirrel_error_map=dp_backward_outputs,
        squirrel_discrim_loss=None,
        squirrel_discrim_grads_and_vars=None,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss=ema.average(gen_loss),
        gen_loss_L1=gen_loss_L1,
        gen_loss_L2=ema.average(gen_loss_L2),
        gen_loss_SSIM=ema.average(gen_loss_SSIM),
        gen_loss_squirrel=None,
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=dp_outputs,
        targets=dp_targets,
        losses={'gen_loss': gen_loss, 'discrim_loss': discrim_loss,
                'gen_loss_L1': gen_loss_L1, 'gen_loss_L2': gen_loss_L2,
                'gen_loss_SSIM': gen_loss_SSIM, 'gen_loss_squirrel': None,
                'squirrel_discrim_loss': None, 'gen_loss_GAN': gen_loss_GAN},
        uncertainty=sigma,
        squirrel_discrim_train=None,
        train=tf.group(update_losses, incr_global_step, gen_train, bw_gen_train, xae_train, yae_train),
    )

def setup_data_loader(data_source, enqueue_data, shuffle=True, batch_size=1, input_size=256, input_channel=1, target_channel=1, repeat=1, control_nc=0, use_mixup=False, seed=123):
    stop_queue_event = threading.Event()
    enqueue_step = batch_size
    maxLen = len(data_source)
    indexes = list(range(maxLen))
    if shuffle:
        random.seed(seed)
        random.shuffle(indexes)
    def enqueue(sess, callback):
        """ Iterates over our data puts small junks into our queue."""
        under = 0

        last_input, last_target = None, None
        try:
            while True:
                if stop_queue_event.is_set():
                    break
                upper = under + enqueue_step

                if upper <= maxLen:
                    dl = [data_source[indexes[i]] for i in range(under, upper)]
                    under = upper
                else:
                    rest = upper - maxLen
                    dl = [data_source[indexes[i]] for i in range(under, maxLen)] + [data_source[indexes[i]] for i in range(0, rest)]
                    under = rest
                curr_input, curr_target, curr_path = zip(*[(d['A'], d['B'], np.array([d['path']])) for d in dl])
                curr_input, curr_target, curr_path = np.stack(curr_input),  np.stack(curr_target), np.stack(curr_path)
                if use_mixup and last_input is not None and last_target is not None:
                    if np.random.random()<0.5:
                        curr_input = curr_input + last_input
                        curr_target = curr_target + last_target
                if use_mixup:
                    last_input = curr_input
                    last_target = curr_target

                curr_control = np.zeros([curr_input.shape[0], 1, 1, control_nc])
                if control_nc > 0:
                    for i, d in enumerate(dl):
                        if 'control' in d and d['control'] is not None:
                            curr_control[i, 0, 0, :] = np.array(d['control'])
                curr_channel_mask = np.ones([curr_input.shape[0], 1, 1, curr_input.shape[3]])
                for i, d in enumerate(dl):
                    if 'channel_mask' in d and d['channel_mask'] is not None:
                        curr_channel_mask[i, 0, 0, :] = np.array(d['channel_mask'])
                assert (not np.any(np.isnan(curr_input)) ) and (not np.any(np.isnan(curr_target)))
                for i in range(repeat):
                    if stop_queue_event.is_set():
                        break
                    enqueue_data(curr_path, curr_input, curr_target, curr_control, curr_channel_mask)
                    print('.', end='')
                    sys.stdout.flush()

                if stop_queue_event.is_set():
                    break
                time.sleep(0)
        except Exception as err:
            import traceback
            print('----------------------enqueue error----------------------')
            traceback.print_exc(file=sys.stdout)
            # print(traceback.format_exception(None, err, err.__traceback__), file=sys.stderr, flush=True)
            print('----------------------enqueue error---------------------->')
            if callback:
                callback()
        finally:
            print("finished enqueueing")
            # coord.request_stop()
            # coord.join(threads)

    # coord = tf.train.Coordinator()
    def queue_start(sess, callback=None):
        # start the threads for our FIFOQueue and batch
        # threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        enqueue_thread = threading.Thread(target=enqueue, args=[sess, callback])
        enqueue_thread.isDaemon()
        enqueue_thread.start()

    def queue_stop():
        stop_queue_event.set()

    return queue_start, queue_stop

def build_network(model_type, input_size, input_nc, output_nc, batch_size, use_resize_conv=False, include_summary=True, ndf=64, ngf=64,
                  gan_weight=1.0, l1_weight=40.0, lr=0.0002, beta1=0.5, lambda_tv=0, norm_A=None, norm_B=None, norm_LR=None,
                  control_nc=0, control_classes=0, lr_nc=0, lr_scale=1, squirrel_weight=20.0, use_gaussd=False, lr_loss_mode='lr_inputs',
                  use_queue=True):
    if norm_A is None:
        norm_A = 'mean_std'
    if norm_B is None:
        norm_B = 'min_max[0,1]'
    if norm_LR is None:
        norm_LR = 'min_max[0,1]'
    assert norm_A == 'mean_std' and (lr_nc is None or lr_nc<=0 or norm_LR == 'min_max[0,1]')
    if use_queue:
        queue_path = tf.placeholder(tf.string, shape=[None, 1], name='queue_path')
        queue_input = tf.placeholder(tf.float32, shape=[None, input_size, input_size, input_nc], name='queue_input')
        queue_target = tf.placeholder(tf.float32, shape=[None, input_size, input_size, output_nc], name='queue_target')
        queue_control = tf.placeholder(tf.float32, shape=[None, 1, 1, control_nc], name='queue_control')
        queue_channel_mask = tf.placeholder(tf.float32, shape=[None, 1, 1, input_nc], name='queue_channel_mask')

        # Build an FIFOQueue
        queue = tf.FIFOQueue(capacity=50, dtypes=[tf.string, tf.float32, tf.float32, tf.float32, tf.float32], shapes=[[1], [input_size, input_size, input_nc], [input_size, input_size, output_nc], [1, 1, control_nc], [1, 1, input_nc]])
        enqueue_op = queue.enqueue_many([queue_path, queue_input, queue_target, queue_control, queue_channel_mask])
        dequeue_op = queue.dequeue()
        close_queue_op = queue.close(cancel_pending_enqueues=True)

        def enqueue_data(sess, curr_path, curr_input, curr_target, curr_control=None, curr_channel_mask=None):
            if curr_control is None and control:
                curr_control = np.zeros([curr_input.shape[0], 1, 1, control_nc])
            if curr_channel_mask is None:
                curr_channel_mask = np.ones([curr_input.shape[0], 1, 1, input_nc])
            sess.run(enqueue_op, feed_dict={'queue_path:0': curr_path,
                                            'queue_input:0': curr_input,
                                            'queue_target:0': curr_target,
                                            'queue_control:0': curr_control,
                                            'queue_channel_mask:0': curr_channel_mask})
        def close_queue(sess):
            sess.run(close_queue_op)

        # tensorflow recommendation:
        # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
        paths_batch, inputs_batch, targets_batch, control_batch, channel_mask_batch = tf.train.batch(dequeue_op, batch_size=batch_size, capacity=40)
    else:
        paths_batch = tf.placeholder(tf.string, shape=[None, 1], name='path')
        inputs_batch = tf.placeholder(tf.float32, shape=[None, input_size, input_size, input_nc], name='input')
        targets_batch = tf.placeholder(tf.float32, shape=[None, input_size, input_size, output_nc], name='target')
        control_batch = tf.placeholder(tf.float32, shape=[None, 1, 1, control_nc], name='control')
        channel_mask_batch = tf.placeholder(tf.float32, shape=[None, 1, 1, input_nc], name='channel_mask')
        enqueue_data = None
        close_queue = None

    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    # inputs and targets are [batch_size, height, width, channels]
    if model_type == 'a_net' or model_type == 'anet':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ms_ssim_l1', use_punet=True, control_nc=control_nc, control_classes=control_classes, use_gaussd=use_gaussd, lr_nc=lr_nc, lr_scale=lr_scale, use_squirrel=lr_nc>0, lr_loss_mode=lr_loss_mode, squirrel_weight=squirrel_weight)
    elif model_type == 'revgan':
        model = create_revgan_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ms_ssim_l1', use_punet=True, control_nc=control_nc, control_classes=control_classes, use_gaussd=use_gaussd, lr_nc=lr_nc, lr_scale=lr_scale, use_squirrel=0, lr_loss_mode=lr_loss_mode, squirrel_weight=0)
    elif model_type == 'unet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv)
    elif model_type == 'punet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_punet=True, control_nc=control_nc, control_classes=control_classes)
    elif model_type == 'ssim_unet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim=True)
    elif model_type == 'ssim_punet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim=True, use_punet=True, control_nc=control_nc, control_classes=control_classes)
    elif model_type == 'ms_ssim_l1_unet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ms_ssim_l1')
    elif model_type == 'ssim_l1_unet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ssim_l1')
    elif model_type == 'ssim_l1_squirrel_unet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ssim_l1', lr_nc=lr_nc, lr_scale=lr_scale, use_squirrel=True, lr_loss_mode=lr_loss_mode, squirrel_weight=squirrel_weight)
    elif model_type == 'ms_ssim_l1_squirrel_unet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ms_ssim_l1', lr_nc=lr_nc, lr_scale=lr_scale, use_squirrel=True, lr_loss_mode=lr_loss_mode, squirrel_weight=squirrel_weight)
    elif model_type == 'ssim_l1_lr_unet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ssim_l1', lr_nc=1, lr_scale=lr_scale)
    elif model_type == 'ssim_l1_punet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ssim_l1', use_punet=True, control_nc=control_nc, control_classes=control_classes)
    elif model_type == 'ssim_l1_squirrel_punet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ssim_l1', use_punet=True, control_nc=control_nc, control_classes=control_classes, lr_nc=lr_nc, lr_scale=lr_scale, use_squirrel=True, lr_loss_mode=lr_loss_mode, squirrel_weight=squirrel_weight)
    elif model_type == 'ms_ssim_l1_squirrel_punet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ms_ssim_l1', use_punet=True, control_nc=control_nc, control_classes=control_classes, lr_nc=lr_nc, lr_scale=lr_scale, use_squirrel=True, lr_loss_mode=lr_loss_mode, squirrel_weight=squirrel_weight)
    elif model_type == 'ssim_l1_fpp_punet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ssim_l1_fpp', use_punet=True, control_nc=control_nc, control_classes=control_classes)
    elif model_type == 'io_ssim_punet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='io_ssim', use_punet=True, control_nc=control_nc, control_classes=control_classes)
    elif model_type == 'io_ssim_unet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='io_ssim')
    elif model_type == 'ms_ssim_unet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ms_ssim')
    elif model_type == 'bayesian_unet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, output_uncertainty=True, bayesian_dropout=True, dropout_prob=dropout_prob, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv)
    elif model_type == 'bayesian_ssim_unet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, output_uncertainty=True, dropout_prob=dropout_prob, bayesian_dropout=True, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim=True)
    elif model_type == 'bayesian_ssim_l1_unet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, output_uncertainty=True, dropout_prob=dropout_prob, bayesian_dropout=True, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ssim_l1')
    elif model_type == 'bayesian_ms_ssim_unet':
        model = create_unet_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, output_uncertainty=True, dropout_prob=dropout_prob, bayesian_dropout=True, use_resize_conv=use_resize_conv, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ms_ssim')
    elif model_type == 'pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_gaussd=use_gaussd)
    elif model_type == 'ssim_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim=True, use_gaussd=use_gaussd)
    elif model_type == 'ms_ssim_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ms_ssim', use_gaussd=use_gaussd)
    elif model_type == 'ms_ssim_l1_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ms_ssim_l1', use_gaussd=use_gaussd)
    elif model_type == 'ssim_l1_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ssim_l1', use_gaussd=use_gaussd)
    elif model_type == 'ssim_l1_squirrel_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ssim_l1', use_gaussd=use_gaussd, lr_nc=lr_nc, lr_scale=lr_scale, use_squirrel=True, lr_loss_mode=lr_loss_mode, squirrel_weight=squirrel_weight)
    elif model_type == 'ssim_l1_punet_squirrel_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ssim_l1', use_punet=True, control_nc=control_nc, control_classes=control_classes, use_gaussd=use_gaussd, lr_nc=lr_nc, lr_scale=lr_scale, use_squirrel=True, lr_loss_mode=lr_loss_mode, squirrel_weight=squirrel_weight)
    elif model_type == 'ms_ssim_l1_squirrel_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ms_ssim_l1', use_gaussd=use_gaussd, lr_nc=lr_nc, lr_scale=lr_scale, use_squirrel=True, lr_loss_mode=lr_loss_mode, squirrel_weight=squirrel_weight)
    elif model_type == 'ms_ssim_l1_punet_squirrel_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ms_ssim_l1', use_punet=True, control_nc=control_nc, control_classes=control_classes, use_gaussd=use_gaussd, lr_nc=lr_nc, lr_scale=lr_scale, use_squirrel=True, lr_loss_mode=lr_loss_mode, squirrel_weight=squirrel_weight)
    elif model_type == 'ssim_l1_punet_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ssim_l1', use_punet=True, control_nc=control_nc, control_classes=control_classes, use_gaussd=use_gaussd)
    elif model_type == 'ms_ssim_l1_punet_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ms_ssim_l1', use_punet=True, control_nc=control_nc, control_classes=control_classes, use_gaussd=use_gaussd)
    elif model_type == 'ssim_l1_fpp_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, bayesian_dropout=False, use_resize_conv=use_resize_conv, gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ssim_l1_fpp', use_punet=True, control_nc=control_nc, control_classes=control_classes, use_gaussd=use_gaussd)
    elif model_type == 'bayesian_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, output_uncertainty=True, bayesian_dropout=True, dropout_prob=dropout_prob, use_resize_conv=use_resize_conv,  gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_gaussd=use_gaussd)
    elif model_type == 'bayesian_ssim_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, output_uncertainty=True, bayesian_dropout=True, dropout_prob=dropout_prob, use_resize_conv=use_resize_conv,  gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim=True, use_gaussd=use_gaussd)
    elif model_type == 'bayesian_ms_ssim_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, output_uncertainty=True, bayesian_dropout=True, dropout_prob=dropout_prob, use_resize_conv=use_resize_conv,  gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ms_ssim', use_gaussd=use_gaussd)
    elif model_type == 'bayesian_ssim_l1_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, output_uncertainty=True, bayesian_dropout=True, dropout_prob=dropout_prob, use_resize_conv=use_resize_conv,  gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ssim_l1', use_gaussd=use_gaussd)
    elif model_type == 'bayesian_ssim_l1_fpp_pix2pix':
        model = create_pix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, output_uncertainty=True, bayesian_dropout=True, dropout_prob=dropout_prob, use_resize_conv=use_resize_conv,  gan_weight=gan_weight, l1_weight=l1_weight, lr=lr, beta1=beta1, lambda_tv=lambda_tv, use_ssim='ssim_l1_fpp', use_punet=True, control_nc=control_nc, control_classes=control_classes, use_gaussd=use_gaussd)

    # elif model_type == 'pix2pix2':
    #     model = create_pix2pix2_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob, use_resize_conv=use_resize_conv)
    # elif model_type == 'fpix2pix':
    #     model = create_fpix2pix_model(inputs_batch, targets_batch, control_batch, channel_mask_batch, ngf=ngf, ndf=ndf, dropout_prob=dropout_prob)
    else:
        raise Exception('unsupported model: ' + model_type)

    inputs = model.inputs
    targets = model.targets
    outputs = model.outputs
    outputs = tf.identity(outputs, name='output')
    if model.squirrel_error_map is not None:
        squirrel_error_map = tf.identity(model.squirrel_error_map, name='error_map')

    if model.lr_inputs is not None:
        lr_inputs = model.lr_inputs
    else:
        lr_inputs = None

    with tf.name_scope("display_fetches"):
        display_fetches = {
            "paths": paths_batch,
            "inputs": inputs, # tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": targets, # tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": outputs, # tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }
        if lr_inputs is not None:
            display_fetches["lr_inputs"] = lr_inputs

        if model.squirrel_error_map is not None:
            display_fetches["squirrel_error_map"] = model.squirrel_error_map

        if model.lr_predict_fake is not None:
            display_fetches["lr_predict_fake"] = model.lr_predict_fake

        if model.lr_predict_real is not None:
            display_fetches["lr_predict_real"] = model.lr_predict_real

        if model.uncertainty is not None:
            display_fetches["aleatoric_uncertainty"] = model.uncertainty # tf.map_fn(tf.image.encode_png, converted_uncertainty, dtype=tf.string, name="uncertainty_png")

    if include_summary:
        def convert(image, scale=False, clip=None):
            channel_count =  image.get_shape()[3]
            #if opt.aspect_ratio != 1.0:
                # upscale to correct aspect ratio
            #    size = [opt.input_size, int(round(opt.input_size * opt.aspect_ratio))]
            #    image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)
            if channel_count > 3:
                image = image[:, :, :, 0:3]
            elif channel_count == 2:
                image = tf.concat([image, tf.zeros([batch_size, input_size, input_size, 1])], axis=3)
            if clip is not None:
                image = tf.clip_by_value(image, clip[0], clip[1])
            if scale:
                image = tf.div( tf.subtract(
                                  image,
                                  tf.reduce_min(image)
                               ),tf.subtract(
                                  tf.reduce_max(image),
                                  tf.reduce_min(image)
                               ))
            return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

        # reverse any processing on images so they can be written to disk or displayed to user
        with tf.name_scope("moving_averaged_loss_fetches"):
            avaraged_loss_fetches = {}
            if model.type == 'GAN':
                avaraged_loss_fetches["discrim_loss"] = model.discrim_loss
                avaraged_loss_fetches["gen_loss_GAN"] = model.gen_loss_GAN
            if model.gen_loss is not None:
                avaraged_loss_fetches["gen_loss"] = model.gen_loss
            if model.gen_loss_L1 is not None:
                avaraged_loss_fetches["gen_loss_L1"] = model.gen_loss_L1
            if model.gen_loss_L2 is not None:
                avaraged_loss_fetches["gen_loss_L2"] = model.gen_loss_L2
            if model.gen_loss_SSIM is not None:
                avaraged_loss_fetches["gen_loss_SSIM"] = model.gen_loss_SSIM
            if model.gen_loss_squirrel is not None:
                avaraged_loss_fetches["gen_loss_squirrel"] = model.gen_loss_squirrel
            if model.squirrel_discrim_loss is not None:
                avaraged_loss_fetches["squirrel_discrim_loss"] = model.squirrel_discrim_loss

        with tf.name_scope("loss_fetches"):
            loss_fetches = {k:v for k, v in model.losses.items() if v is not None}

        # summaries
        with tf.name_scope("inputs_summary"):
            tf.summary.image("inputs", convert(inputs, scale=True))

        if lr_inputs is not None:
            with tf.name_scope("lr_inputs_summary"):
                tf.summary.image("lr_inputs", convert(lr_inputs, scale=True))

        with tf.name_scope("targets_summary"):
            tf.summary.image("targets", convert(targets))

        with tf.name_scope("outputs_summary"):
            tf.summary.image("outputs", convert(outputs))

        if model.uncertainty is not None:
            with tf.name_scope("uncertainty_summary"):
                tf.summary.image("uncertainty", convert(model.uncertainty, scale=True))

        if model.predict_real is not None:
            with tf.name_scope("predict_real_summary"):
                tf.summary.image("predict_real", convert(model.predict_real, scale=True))

        if model.predict_fake is not None:
            with tf.name_scope("predict_fake_summary"):
                tf.summary.image("predict_fake", convert(model.predict_fake, scale=True))

        if model.discrim_loss is not None:
            tf.summary.scalar("discriminator_loss", model.discrim_loss)

        if model.gen_loss_GAN is not None:
            tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)

        if model.lr_predict_fake is not None:
            with tf.name_scope("lr_predict_fake_summary"):
                tf.summary.image("lr_predict_fake", convert(model.lr_predict_fake, scale=True, clip=(0, 1)))

        if model.lr_predict_real is not None:
            with tf.name_scope("lr_predict_real_summary"):
                tf.summary.image("lr_predict_real", convert(model.lr_predict_real, scale=True, clip=(0, 1)))

        if model.squirrel_error_map is not None:
            with tf.name_scope("squirrel_error_map_summary"):
                tf.summary.image("squirrel_error_map", convert(model.squirrel_error_map, clip=(0, 1)))

        if model.gen_loss_squirrel is not None:
            tf.summary.scalar("generator_loss_squirrel", model.gen_loss_squirrel)

        if model.squirrel_discrim_loss is not None:
            tf.summary.scalar("squirrel_discrim_loss", model.squirrel_discrim_loss)

        if model.gen_loss is not None:
            tf.summary.scalar("generator_loss", model.gen_loss)

        if model.gen_loss_L1 is not None:
            tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

        if model.gen_loss_L2 is not None:
            tf.summary.scalar("generator_loss_L2", model.gen_loss_L2)

        if model.gen_loss_SSIM is not None:
            tf.summary.scalar("generator_loss_SSIM", model.gen_loss_SSIM)

        '''for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)

        _grads_and_vars = model.gen_grads_and_vars
        if model.discrim_grads_and_vars is not None:
            _grads_and_vars = _grads_and_vars + model.discrim_grads_and_vars
        if model.squirrel_discrim_grads_and_vars is not None:
            _grads_and_vars = _grads_and_vars + model.squirrel_discrim_grads_and_vars
        for grad, var in _grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad)'''

        summary_merged = tf.summary.merge_all()
    else:
        summary_merged = None
    return model, (enqueue_data, close_queue), display_fetches, (loss_fetches, avaraged_loss_fetches), summary_merged


def export(sess, output_dir):
    export_saver = tf.train.Saver()
    print("exporting model")
    export_saver.export_meta_graph(filename=os.path.join(output_dir, "export.meta"))
    export_saver.save(sess, os.path.join(output_dir, "export"), write_meta_graph=False)


def export_visualization(visualizeDict, output_dir):
    from tensorflow.contrib.tensorboard.plugins import projector
    latent_vectors = visualizeDict['latent_vectors']
    latent_vector_labels = visualizeDict['latent_vector_labels']
    image_paths = visualizeDict['image_paths']

    LOG_DIR = os.path.join(output_dir, 'visualize')
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    print('saving latent vectors...')
    metadata_file = open(os.path.join(LOG_DIR, 'metadata.tsv'), 'w')
    metadata_file.write('Index\tPATH\tROW\tCOL\n')

    image_dir = os.path.join(output_dir, "outputs")
    img_data=[]
    a, b, _ = latent_vectors[0].shape
    for ind, p in enumerate(image_paths):
        input_img=scipy.misc.imread(os.path.join(image_dir, os.path.splitext(p)[0]+'-targets.png'))
        w, h = input_img.shape[0]//a, input_img.shape[1]//b
        for j in range(b):
            for i in range(a):
                input_img_resize=scipy.misc.imresize(scipy.misc.bytescale(input_img[j*h:(j+1)*h, i*w:(i+1)*w])*10, (224,224))
                img_data.append(input_img_resize)
                metadata_file.write('%06d\t%s\t%d\t%d\n' % (ind, latent_vector_labels[ind], i, j))
    img_data = np.array(img_data)
    sprite = images_to_sprite(img_data)
    scipy.misc.imsave(os.path.join(LOG_DIR, 'sprite.png'), sprite)

    metadata_file.close()

    tf.reset_default_graph()
    with tf.Session() as sess:
        EMB = np.concatenate([v.reshape(-1, v.shape[-1]) for v in latent_vectors], axis=0)
        print(EMB.shape, len(latent_vector_labels))
        embedding_var = tf.Variable(EMB,  name='neck_of_unet')
        sess.run(embedding_var.initializer)
        summary_writer = tf.summary.FileWriter(LOG_DIR)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
        embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite.png')
        embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
        projector.visualize_embeddings(summary_writer, config)
        saver = tf.train.Saver([embedding_var])
        saver.save(sess, os.path.join(LOG_DIR, 'model2.ckpt'), 1)
        print('latent vector saved to ', LOG_DIR)

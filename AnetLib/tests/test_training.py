import os
import sys
import tensorflow as tf
from AnetLib.options.train_options import TrainOptions
from AnetLib.models.models import create_model
from smlm_datasets import create_data_sources

def test_training():
    opt = TrainOptions().parse(['--workdir=./__test_tmp__/'])
    opt.model = 'a_net_tensorflow'
    opt.fineSize = 256
    opt.batchSize = 1
    opt.dim_ordering = 'channels_last'
    opt.display_freq = 500
    opt.use_resize_conv = True
    opt.norm_A = 'mean_std'
    opt.norm_B = 'min_max[0,1]'
    opt.lambda_A = 50
    opt.input_nc = 2
    opt.lr_nc = 1
    opt.lr_scale = 1.0/4.0
    opt.lambda_LR = 0
    opt.control_nc = 1
    opt.add_data_type_control = True
    opt.add_lr_channel = 'pseudo'
    # opt.continue_train = True

    # start training
    sources = create_data_sources(['TransformedTubulin001NB'], opt)
    d = sources['train']
    model = create_model(opt)
    model.train(d, verbose=1, max_epoch=1)

    # training done
    opt.phase = 'test'
    model = create_model(opt)
    sources = create_data_sources(['TransformedTubulin001NB'], opt)
    d = sources['test']
    model.predict(d, verbose=1)

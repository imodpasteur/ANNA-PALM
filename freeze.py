#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Freeze A-Net models
python3 freeze.py --workdir=./results/frozen_model_1 --load_dir=./results/simulated_model
'''
import os
import sys
import tensorflow as tf
from AnetLib.options.train_options import Options
from AnetLib.models.models import create_model
from smlm_datasets import create_data_sources
from AnetLib.util.freeze_graph import freeze_latest_checkpoint

default_workdir = './workdir'
opt = Options().parse()
opt.model = 'anet_tensorflow'
opt.fineSize = 2560
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
opt.continue_train = True
opt.no_queue = True

model = create_model(opt)
model.save('latest')
model.close()



freeze_latest_checkpoint(os.path.join(opt.workdir, '__model__'), output_file=os.path.join(opt.workdir, "tensorflow_model.pb"), output_node_names='output,error_map')

print('freezed model saved: ' + os.path.join(opt.workdir, "tensorflow_model.pb"))

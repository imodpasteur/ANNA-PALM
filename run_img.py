import os, sys
import tensorflow as tf
from AnetLib.options.train_options import Options
from AnetLib.models.models import create_model
from smlm_datasets import create_data_sources

default_workdir = './output/' + os.path.basename(sys.argv[0])
opt = Options().parse()
opt.fineSize = 512
opt.batchSize = 1  # batchSize = 1
opt.model = 'a_net_tensorflow'
opt.dim_ordering = 'channels_last'
opt.display_freq = 500
opt.save_latest_freq = 1000
opt.use_resize_conv = True
opt.norm_A = 'mean_std'
opt.norm_B = 'min_max[0,1]'
opt.lambda_A = 50
opt.input_nc = 2
opt.lr_nc = 1
opt.lr_scale = 1.0/4.0
opt.lambda_LR = 25
opt.control_nc = 1
opt.add_data_type_control = True
opt.add_lr_channel = False
opt.use_random_channel_mask = True
opt.lr_loss_mode = 'lr_predict'


sources = create_data_sources('TransformedGenericImages', opt)
d = sources['train']
# noise_source = create_data_sources('NoiseCollection001', opt)['train']
# d.set_addtional_source(noise_source)
model = create_model(opt)
model.train(d, verbose=1, max_steps=200000)

opt.phase = 'test'
opt.fineSize = 1024
model = create_model(opt)
sources = create_data_sources('TransformedGenericImages', opt)
d = sources['test']
model.predict(d, verbose=1)

import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.isTrain = False

    def initialize(self):
        self.parser.add_argument('--workdir', type=str, required=True, help='work directory')
        self.parser.add_argument('--name', type=str, default=None, help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--load_dir', type=str, default=None, help='load weights from path')
        self.parser.add_argument('--save_dir', type=str, default=None, help='path for save outputs and configs')
        self.parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
        self.parser.add_argument('--dataroot', type=str, default=None, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--which_network', type=str, default=None, help='which network to be loaded(all, G, D, S)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        self.parser.add_argument('--control_nc', type=int, default=0, help='# of channels for control sliders')
        self.parser.add_argument('--control_classes', type=int, default=None, help='# of classes to be encoded for the 1st control slider channel')
        self.parser.add_argument('--lr_nc', type=int, default=0, help='channel number for low-res input image')
        self.parser.add_argument('--lr_scale', type=float, default=1.0/4.0, help='scale factor for low-res image')
        self.parser.add_argument('--lr_sigma', type=int, default=8,  help='the gaussian blurring simga used to generate pseudo low resolution image')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='unet_256', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model', type=str, default='a_net_tensorflow',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default=None, help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--use_resize_conv', action='store_true', help='if specified, use resize convolution instead of deconvolution')
        self.parser.add_argument('--dim_ordering', type=str, default=None, help='dim ordering, for tensorflow, should be channels_last, for pytorch should be channels_first')
        self.parser.add_argument('--norm_A', type=str, default=None, help='normalization method for input image')
        self.parser.add_argument('--norm_B', type=str, default=None, help='normalization method for target image')
        self.parser.add_argument('--norm_LR', type=str, default=None, help='normalization method for low-res image')
        self.parser.add_argument('--add_data_type_control', action='store_true', help='if specified, add dataset type channel to control sliders')
        self.parser.add_argument('--add_fpp_control', action='store_true', help='if specified, add false-positive-prevention channel to control sliders')
        self.parser.add_argument('--add_lr_channel', type=str, default=None, help='if specified, add low-res channel with pseudo or empty mode')
        self.parser.add_argument('--no_config_override', action='store_true', help='if specified, configurations are not allowed to overried from config.json file')
        self.parser.add_argument('--use_mixup', action='store_true', help='if specified, using mixup data augmentation')
        self.parser.add_argument('--use_random_channel_mask', action='store_true', help='if specified, using channels will be randomly removed')
        self.parser.add_argument('--use_gaussd', action='store_true', help='if specified, using gaussian blur before feed to the discriminator')
        self.parser.add_argument('--no_queue', action='store_true', help='if specified, disable data queue, replace with only placeholder')
        self.initialized = True

    def parse(self, *args):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args(*args)
        self.opt.isTrain = self.isTrain   # train or test

        # clean up all the space
        if self.opt.norm_A is not None:
            self.opt.norm_A = self.opt.norm_A.replace(' ', '')
        if self.opt.norm_B is not None:
            self.opt.norm_B = self.opt.norm_B.replace(' ', '')
        if self.opt.norm_LR is not None:
            self.opt.norm_LR = self.opt.norm_LR.replace(' ', '')

        if self.opt.dim_ordering is None and 'tensorflow' in self.opt.model:
            self.opt.dim_ordering = 'channels_last'

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if not os.path.exists(self.opt.workdir):
            os.makedirs(self.opt.workdir)

        if self.opt.save_dir is None:
            if self.opt.name:
                self.opt.save_dir = os.path.join(self.opt.workdir, self.opt.name)
            else:
                self.opt.save_dir = os.path.join(self.opt.workdir, 'outputs')
            if not os.path.exists(self.opt.save_dir):
                os.makedirs(self.opt.save_dir)

        if self.opt.checkpoints_dir is None:
            self.opt.checkpoints_dir = os.path.join(self.opt.workdir, '__model__')
            if not os.path.exists(self.opt.checkpoints_dir):
                os.makedirs(self.opt.checkpoints_dir)

        if self.opt.load_dir is None:
            self.opt.load_dir = self.opt.checkpoints_dir
            if os.path.exists(os.path.join(self.opt.load_dir, '__model__')):
                self.opt.load_dir = os.path.join(self.opt.load_dir, '__model__')

        if self.opt.dataroot is None:
            self.opt.dataroot = self.opt.workdir

        return self.opt

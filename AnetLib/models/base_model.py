import os
import time
import torch
from ..data.flex_data_loader import CreateDataLoader
from PIL import Image
import AnetLib.util.util as util
from torch.utils.data.dataloader import default_collate
import json
import argparse

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        # set gpu ids
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        if 'dim_ordering' in opt:
            assert opt.dim_ordering == 'channels_first'
        else:
            opt.dim_ordering = 'channels_first'
        self.Tensor = torch.cuda.FloatTensor if len(self.gpu_ids) and torch.cuda.is_available() else torch.Tensor

        assert self.opt.which_direction == 'AtoB'

        self._current_report = {}
        self._current_visuals = {}
        self._current_config = vars(opt).copy()
        self._current_epoch = 0

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def initialize_from_config(self, config_path, **kwargs):
        with open(config_path, 'r') as f:
            config_json = json.load(f)
            parser = argparse.ArgumentParser()
            opt = parser.parse_args([])
            config = vars(opt)
            config.update(config_json)
            config.update(kwargs)
            self.initialize(opt)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self, dropout=0):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        images = {}
        for k, v in self._current_visuals.items():
            images[k] = util.tensor2im(v.data)
        return images

    def get_current_report(self):
        report = {}
        for k, v in self._current_report.items():
            report[k] = v.data[0]
        return report

    def save_current_visuals(self, label=None):
        images = self.get_current_visuals()
        if 'path' in self.input:
            output_paths = [os.path.join(self.opt.save_dir, os.path.split(p)[1]) for p in self.input['path']]
        else:
            output_paths = [self.opt.save_dir for i in range(len(images))]
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)
        for k, v in images.items():
            for b in range(v.shape[0]):
                ima = v[b]
                channels = ima.shape[2]
                for i in range(channels):
                    im = Image.fromarray(ima[:, :, i])
                    d, n = os.path.split(output_paths[b])
                    n = '{}_{}_b{}_i{}.tif'.format(n, k, b, i)
                    if label:
                        n = '{}_'.format(label) + n
                    im.save(os.path.join(d, n))

    def save(self, label):
        self.save_config(label)

    def load(self, label):
        self.load_config(label)

    def save_config(self, label):
        self._current_config['_current_epoch'] = self._current_epoch
        opt = vars(self.opt)
        for k in opt:
            self._current_config[k] = opt[k]

        save_filename = '%s_config.json' % (label)
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        with open(save_path, 'w') as f:
            json.dump(self._current_config, f)
        print('config({})  saved to {}'.format(label, save_path))

    def load_config(self, label):
        load_filename = '%s_config.json' % (label)
        load_path = os.path.join(self.opt.checkpoints_dir, load_filename)
        if os.path.exists(load_path):
            with open(load_path, 'r') as f:
                config_json = json.load(f)
                # print('config({}) loaded:'.format(label))
                # print('-------')
                # for k, v in sorted(config_json.items()):
                #     if k not in self._current_config or self._current_config[k] != v:
                #         print('%s: %s' % (str(k), str(v)))
                # print('-------')
                for k in config_json.keys():
                    self._current_config[k] = config_json[k]
        self._current_epoch = self._current_config.get('_current_epoch', self._current_epoch)

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device_id=gpu_ids[0])
        print('network {} saved to {}'.format(network_label, save_path))

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        network.load_state_dict(torch.load(save_path))
        print('network {} loaded from {}'.format(network_label, save_path))

    def update_learning_rate():
        pass

    def train(self, data_source_train, data_source_test=None, epoch_callback=None, batch_callback=None, transform=None, cached=False, verbose=1):
        data_loader = CreateDataLoader(data_source_train, self.opt, cached=cached, transform=transform, verbose=verbose)
        dataset = data_loader.load_data()
        dataset_size = len(data_loader)
        print('#training images = %d' % dataset_size)

        if data_source_test:
            data_loader_test = CreateDataLoader(data_source_test, self.opt, cached=cached, verbose=verbose-1)
            dataset_test = data_loader_test.load_data()
            dataset_test_size = len(data_loader_test)
            print('#test images = %d' % dataset_test_size)
        else:
            dataset_test = dataset
            dataset_test_size = dataset_size
            print('WARNING: using training set as test set.')

        step = 0
        start = self._current_epoch
        for epoch in range(1 + start, self.opt.niter + self.opt.niter_decay + 1 + start):
            epoch_start_time = time.time()
            self.set_input(next(iter(dataset_test)))
            self.test()
            self.save_current_visuals(label=str(epoch))
            test_report = self.get_current_report()
            if verbose > 0:
                print('<== Epoch {}, test-> '.format(epoch)+', '.join(['{}:{:.5f}'.format(k,v) for k,v in test_report.items()]))
            for i, data in enumerate(dataset):
                iter_start_time = time.time()
                step += self.opt.batchSize
                epoch_iter = step - dataset_size * (epoch - 1)
                self.set_input(data)
                self.optimize_parameters()
                if step % self.opt.print_freq == 0:
                    train_report = self.get_current_report()
                    print('\n', ', '.join(['{}:{:.5f}'.format(k,v) for k,v in train_report.items()]))
                if step % self.opt.save_latest_freq == 0:
                    if verbose > 0:
                        print('\nsaving the latest model (epoch {}, step {})'.format(epoch, step))
                    self.save('latest')
                if batch_callback:
                    try:
                        details = {'epoch': epoch, 'step': step, 'test_report': test_report, 'train_report': train_report}
                        batch_callback(self, details)
                    except Exception as e:
                        print('\nerror in batch callback.')
                if verbose > 1:
                    print('.', end='')
            if epoch % self.opt.save_epoch_freq == 0:
                if verbose > 0:
                    print('saving the model at the end of epoch {}, iters {}'.format(epoch, step))
                self.save('latest')
                self.save(epoch)
            if epoch > self.opt.niter:
                self.update_learning_rate()
            self._current_epoch = epoch
            if verbose > 0:
                print('\n==> Epoch {} / {} \t Time Taken: {} sec'.format(epoch, self.opt.niter + self.opt.niter_decay, time.time() - epoch_start_time))
            if epoch_callback:
                try:
                    details = {'epoch': epoch, 'step': step, 'test_report': test_report, 'train_report': train_report}
                    epoch_callback(self, details)
                except Exception as e:
                    print('error in epoch callback.')

    def predict(self, data_source, dropout=0, cached=False, verbose=1):
        data_loader_test = CreateDataLoader(data_source, self.opt, cached=cached, verbose=verbose)
        dataset_test = data_loader_test.load_data()
        dataset_test_size = len(data_loader_test)
        if verbose > 0:
            print('#test images = %d' % dataset_test_size)

        for i, data in enumerate(dataset_test):
            self.set_input(data)
            self.test(dropout=dropout)
            self.save_current_visuals(label=str(i))
            if verbose > 0:
                report = self.get_current_report()
                print('{}: processed {}, batch {}, size {}x{}'.format(i, data['A_paths'], data['A'].size(0), data['A'].size(2), data['A'].size(3)), end='')
                print('  ('+', '.join(['{}:{:.4f}'.format(k,v) for k,v in report.items()])+')')

    def train_image(self, img_dict):
        self.set_input(default_collate([img_dict]))
        self.optimize_parameters()

    def train_batch(self, batch_dict):
        self.set_input(default_collate(img_dict))
        self.optimize_parameters()

    def predict_image(self, img_dict, dropout=0):
        self.set_input(default_collate([img_dict]))
        self.test(dropout=dropout)

    def predict_batch(self, img_dict, dropout=0):
        self.set_input(default_collate(img_dict))
        self.test(dropout=dropout)

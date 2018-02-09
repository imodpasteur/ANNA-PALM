import os
import threading
import time
import random
import torch.utils.data
from .base_data_loader import BaseDataLoader
from .image_folder import ImageFolder
from pdb import set_trace as st
# pip install future --upgrade
from builtins import object


def CreateDataLoader(dataset, opt, cached=False, transform=None, verbose=1):
    data_loader = None
    data_loader = FlexDataLoader()
    data_loader.initialize(dataset, opt, cached=cached, transform=transform, verbose=verbose)
    return data_loader


class PairedData(object):
    def __init__(self, data_loader, max_dataset_size, dataroot, transform=None):
        self.data_loader = data_loader
        self.max_dataset_size = max_dataset_size
        self.dataroot = dataroot
        self.transform = transform
        # st()

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration

        D = next(self.data_loader_iter)
        if self.transform:
            return self.transform(D)
        else:
            return D


class CachedPairedData(object):
    def __init__(self, data_loader, max_dataset_size, dataroot, cache_len=100, preload_len=1, transform=None, verbose=1):
        self.data_loader = data_loader
        self.max_dataset_size = max_dataset_size
        self.dataroot = dataroot
        self.cache_len = cache_len
        self.preload_len = preload_len
        assert cache_len >= preload_len
        self.__cache = []
        self.__cache_count = 0
        self.__last_cache_count = 0
        self.verbose = verbose
        self.__threadErr = None
        self.__abort = None
        self.__threadAborted = threading.Event()
        self.data_loader_length = len(self.data_loader)
        self.transform = transform
        # st()

    def _preload_data(self, data_iter, preload_len):
        if self.verbose > 0:
            print('preloading')
        for i in range(preload_len):
            if self.verbose > 2:
                print(i, end='')
            D = next(data_iter)
            self.__cache.append(D)
        if self.verbose > 0:
            print('preload done, data length: {}'.format(preload_len))

    def _fill_data(self, data_iter, cache_len, abort):
        if self.verbose > 1:
            print('starting fill_data thread')
        while True:
            try:
                if abort.isSet():
                    if self.verbose > 1:
                        print('quiting fill_data thread')
                    break
                if self.__last_cache_count + self.cache_len <= self.__cache_count:
                    # skip loading if cached data is not used for a while
                    time.sleep(0.5)
                    if self.verbose > 0:
                        print('|', end='')
                    continue
                D = next(data_iter)
                # if self.verbose > 1:
                #     print('filling ', D.keys())
                self.__cache.append(D)
                self.__cache_count += 1
                if len(self.__cache) > cache_len:
                    # if self.verbose > 1:
                    #     print('poping ', self.__cache[0].keys())
                    self.__cache.pop(0)
            except StopIteration:
                # self.__threadErr = StopIteration
                # self.__threadAborted.set()
                data_iter = iter(self.data_loader)
                if self.verbose > 1:
                    print('restart data loader')
            except Exception as e:
                self.__threadErr = e
                self.__threadAborted.set()
                raise e


    def __iter__(self):
        self.iter = 0
        if hasattr(self, 'data_loader_iter') and self.data_loader_iter and not self.__threadAborted.isSet():
            return self

        self.data_loader_iter = iter(self.data_loader)
        self.data_loader_length = len(self.data_loader)
        self._preload_data(self.data_loader_iter, self.preload_len)
        if self.__abort:
            self.__abort.set()
        self.__abort = threading.Event()
        self.__thread = threading.Thread(name='fill_data_thread', target=self._fill_data, args=(self.data_loader_iter, self.cache_len, self.__abort, ))
        self.__thread.daemon = True
        self.__threadAborted.clear()
        self.__thread.start()
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            self.__abort.set()
            raise StopIteration
        if self.__threadAborted.isSet():
            if self.__threadErr:
                raise self.__threadErr
            else:
                print('WARNINIG: data loader thread aborted.')
        if self.iter >= self.data_loader_length:
            self.iter = 0
            raise StopIteration
        D = random.choice(self.__cache).copy() #next(self.data_loader_iter)
        if self.verbose > 0:
            print(self.__cache_count-self.__last_cache_count, end=',')
        self.__last_cache_count = self.__cache_count
        if self.transform:
            return self.transform(D)
        else:
            return D


class FlexDataLoader(BaseDataLoader):
    def initialize(self, dataset, opt, cached=False, transform=None, verbose=1):
        BaseDataLoader.initialize(self, opt)
        self.fineSize = opt.fineSize
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))

        self.dataset = dataset
        if cached:
            self.paired_data = CachedPairedData(data_loader, opt.max_dataset_size, opt.dataroot, cache_len=len(dataset), transform=transform, verbose=verbose)
        else:
            self.paired_data = PairedData(data_loader, opt.max_dataset_size, opt.dataroot, transform=transform)

    def name(self):
        return 'FlexDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

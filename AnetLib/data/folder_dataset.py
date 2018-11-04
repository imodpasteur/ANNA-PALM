################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################
import os
import os.path
import fnmatch
import numpy as np
from collections import OrderedDict

def make_dataset(rootdir, channels, recursive=True, allow_unpaired=False):
    images = []
    assert os.path.isdir(rootdir), '%s is not a valid directory' % rootdir
    files = {}

    loaders = {}
    for ch, v in channels.items():
        files[ch] = []
        if recursive:
            for root, _, fnames in sorted(os.walk(rootdir)):
                for fname in fnames:
                    if fnmatch.fnmatch(fname, v['filter']) and not fname.startswith('.'):
                        path = os.path.join(root, fname)
                        files[ch].append(path)
        else:
            for fname in sorted(os.listdir(rootdir)):
                if fnmatch.fnmatch(fname, v['filter']) and not fname.startswith('.'):
                    path = os.path.join(rootdir, fname)
                    files[ch].append(path)
    # removed unpaired files
    valid_files = []
    k, v = list(channels.items())[0]
    id0 = v['filter'].replace('*', '').split('.')[0]

    for f in files[k]:
        mark = True
        for ch, v in channels.items():
            id = v['filter'].replace('*', '').split('.')[0]
            ch_f = f.replace(id0, id)
            if not ch_f in files[ch]:
                mark = False
                break

        if mark or allow_unpaired:
            valid_channels = {}
            for ch, v in channels.items():
                id = v['filter'].replace('*', '').split('.')[0]
                ch_f = f.replace(id0, id)
                valid_channels[ch] = ch_f
            valid_files.append(valid_channels)

    # cache files
    for fs in valid_files:
        for ch in channels:
            channels[ch]['loader'].cache(fs[ch])

    for ch in channels:
        channels[ch]['loader'].save_cache(rootdir)

    return valid_files

class FolderDataset(): # data.Dataset):

    def __init__(self, root, channels={}, transform=None, repeat=1, recursive=True, allow_unpaired=False):
        self.channels = channels
        self.repeat = repeat
        files = make_dataset(root, channels, recursive, allow_unpaired)
        if len(files) == 0:
            raise(RuntimeError("Found 0 samples in: " + root))
        print('Found {} samples in {}.'.format(len(files), root))
        self.root = root
        self.files = files
        self.transform = transform
        self.__index = None
        self.__data = None

    def __getitem__(self, index):
        i = (index//self.repeat) % len(self.files)
        if self.__index and self.__index == i:
            ret = self.__data.copy()
        else:
            paths = self.files[i]
            ret = {}
            for ch, path in paths.items():
                ret[ch] = self.channels[ch]['loader'](path)
                ret[ch+'.path'] = path
            self.__data = ret.copy()
            self.__index = i
        for ch, path in self.files[i].items():
            ret[ch+'.repeat'] = index%self.repeat
        if self.transform is not None:
            ret = self.transform(ret.copy())
        return ret

    def __len__(self):
        return self.repeat * len(self.files)


def make_subfolder_dataset(rootdir, channels, allow_unpaired=False):
    images = []
    assert os.path.isdir(rootdir), '%s is not a valid directory' % rootdir

    samples = []
    for subfolder in sorted(os.listdir(rootdir)):
        samplefolder = os.path.join(rootdir, subfolder)
        if not os.path.isdir(samplefolder): continue
        files = {}
        for ch, v in channels.items():
            chfiles = []
            for fname in sorted(os.listdir(samplefolder)):
                if fnmatch.fnmatch(fname, v['filter']) and not fname.startswith('.'):
                    path = os.path.join(samplefolder, fname)
                    chfiles.append(path)
            if len(chfiles) > 0:
                files[ch] = chfiles
        if len(files.keys()) == len(channels.items()):
            files['__path__'] = samplefolder
            samples.append(files)
    return samples


class SubfolderDataset(): # data.Dataset):

    def __init__(self, root, channels=[], transform=None, repeat=1, allow_unpaired=False):
        self.channels = OrderedDict(channels)
        self.repeat = repeat
        files = make_subfolder_dataset(root, self.channels, allow_unpaired)
        if len(files) == 0:
            raise(RuntimeError("Found 0 samples in: " + root))
        print('Found {} samples in {}.'.format(len(files), root))
        self.root = root
        self.files = files
        self.transform = transform
        self.__index = None
        self.__data = None

    def __getitem__(self, index):
        i = (index//self.repeat) % len(self.files)
        if self.__index and self.__index == i:
            ret = self.__data.copy()
        else:
            paths = self.files[i]
            ret = {}
            for ch, path in paths.items():
                if ch == '__path__':
                    ret[ch] = path
                else:
                    if type(path) is list:
                        if len(path) == 1:
                            path = path[0]
                        else:
                            path = random.choice(path)
                    ret[ch] = self.channels[ch]['loader'](path)
                    ret[ch+'.path'] = path

            self.__data = ret.copy()
            self.__index = i
        for ch, path in self.files[i].items():
            ret[ch+'.repeat'] = index%self.repeat
        if self.transform is not None:
            ret = self.transform(ret.copy())
        return ret

    def __len__(self):
        return self.repeat * len(self.files)

if __name__ == '__main__':
    channels = {'cells': {'filter':'*Cy5.tif', 'loader':ImageLoader()},
                'nulcei': {'filter':'*DAPI.tif', 'loader':ImageLoader()},
                'cell_dist_mask': {'filter':'*Cy5_ROI.zip', 'loader':ImageJRoi2DistanceMap(2048)},
                'cell_mask': {'filter':'*Cy5_ROI.zip', 'loader':ImageJRoi2Mask(2048)},
                'nuclei_mask':{'filter':'*DAPI_ROI.zip', 'loader':ImageJRoi2Edge(2048)}}

    dataset = FolderDataset('../florian_cell_mask_v0.1.1/train',
                    channels = channels)

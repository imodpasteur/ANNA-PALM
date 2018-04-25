import os
import numpy as np
import hashlib
import json
import scipy
import fnmatch

from skimage import io


def check_integrity(fpath, md5c):
    if not os.path.isfile(fpath):
        return False
    md5 = hashlib.md5(open(fpath, 'rb').read()).hexdigest()
    if md5 != md5c:
        print('intergrity check failed: md5='+ md5)
        return False
    print('intergrity check passed: md5='+ md5)
    return True


def download_url(url, fpath, md5c=None):
    from six.moves import urllib
    # downloads file
    if md5c and os.path.isfile(fpath) and hashlib.md5(open(fpath, 'rb').read()).hexdigest() == md5c:
        print('Using downloaded file: ' + fpath)
    else:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(url, fpath)
    print('Done!')


def get_id_for_dict(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode('utf-8')).hexdigest()


class RepeatDataset():
    def __init__(self, dataset, repeat, transform, overwrite_length=None):
        self.dataset = dataset
        self.repeat = repeat
        self.transform = transform
        self.overwrite_length = overwrite_length
        if self.overwrite_length:
            assert len(self.dataset) >= self.overwrite_length

    def __getitem__(self, index):
        i = index // self.repeat
        transform = self.transform
        if type(self.transform) is list:
            assert len(self.transform) == self.repeat, 'length of transform list must equal to repeat times'
            j = index % self.repeat
            transform = self.transform[j]

        ret = self.dataset[i]
        if transform is not None:
            ret = transform(ret)

        return ret

    def __len__(self):
        if self.overwrite_length:
            return self.overwrite_length * self.repeat
        else:
            return len(self.dataset) * self.repeat

class NpzDataset():
    url = "https://www.dropbox.com/s/7v5vmgc3qkd2901/tubulin-sim-800x800.npz?dl=1"
    file_name = 'tubulin-sim-800x800.npz'
    npz_md5 = 'e2ae7b3d736a4c72479281df054c1eaf'

    def __init__(self, root, train=True, transform=None, download=False, ratio=0.95, repeat=1):
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.repeat = repeat
        self._last_ = None

        if not os.path.exists(root):
            os.makedirs(root)
        self.fpath = os.path.join(root, self.file_name)

        if download:
            download_url(self.url, self.fpath, self.npz_md5)
        if not check_integrity(self.fpath, self.npz_md5):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        print(self.fpath)

        data = np.load(self.fpath)
        X_, y_ = data['y'], data['y']
        X_, y_ = X_.transpose((0, 2, 3, 1)), y_.transpose((0, 2, 3, 1))
        X_, y_ = X_.astype('float32'), y_.astype('float32')
        train = int(ratio * X_.shape[0])
        self.X_train, self.y_train = X_[:train], y_[:train]
        self.X_test, self.y_test = X_[train:], y_[train:]

    def __getitem__(self, index):
        i = index // self.repeat
        if self._last_ and self._last_[0] == i:
            ret = self._last_[1]
            return ret

        if self.train:
            img, target = self.X_train[i], self.y_train[i]
        else:
            img, target = self.X_test[i], self.y_test[i]
        ret = (img, target)
        if self.transform is not None:
            ret = self.transform([img, target])

        self._last_ = (i, ret)
        return ret

    def __len__(self):
        if self.train:
            return self.X_train.shape[0] * self.repeat
        else:
            return self.X_test.shape[0] * self.repeat

    def calculate_mean_std(self):
        self.X_mean = self.X_train.mean(axis=(0,1,2))
        self.X_std = self.X_train.std(axis=(0,1,2))
        self.y_mean = self.y_train.mean(axis=(0,1,2))
        self.y_std = self.y_train.std(axis=(0,1,2))
        return self.X_mean, self.X_std, self.y_mean, self.y_std

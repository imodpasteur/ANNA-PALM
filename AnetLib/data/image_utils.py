import numpy as np
import scipy
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import collections
from PIL import Image
import numbers
import os
from .file_loader import FileLoader
__author__ = "Wei OUYANG"
__license__ = "GPL"
__version__ = "0.1.0"
__status__ = "Development"


def center_crop(x, center_crop_size):
    assert x.ndim == 3
    centerh, centerw = x.shape[1] // 2, x.shape[2] // 2
    halfh, halfw = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[:, centerh - halfh:centerh + halfh, centerw - halfw:centerw + halfw]


def to_numpy(x):
    x = x.numpy()
    return x.transpose((1, 2, 0))


def is_numeric(obj):
    attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)


def num_generator(config, index=0, random_state=np.random):
    if config[0] == 'uniform':
        ret = random_state.uniform(config[1], config[2], 1)[0]
    elif config[0] == 'lognormal':
        ret = random_state.lognormal(config[1], config[2], 1)[0]
    elif is_numeric(config[0]):
        ret = config[index]
    else:
        print(config)
        raise Exception('unsupported format')
    return ret


def poisson_downsampling(image, peak, random_state=np.random):
    if not isinstance(image, np.ndarray):
        imgArr = np.array(image, dtype='float32')
    else:
        imgArr = image.astype('float32')
    Q = imgArr.max(axis=(0, 1)) / peak
    if Q[0] == 0:
        return imgArr
    ima_lambda = imgArr / Q
    noisy_img = random_state.poisson(lam=ima_lambda)
    return noisy_img.astype('float32')


def elastic_transform(image, alpha=1000, sigma=30, spline_order=1, mode='nearest', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert image.ndim == 3
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
    return result


class Merge(object):
    """Merge a group of images
    """

    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, images):
        if isinstance(images, collections.Sequence) or isinstance(images, np.ndarray):
            assert all([isinstance(i, np.ndarray)
                        for i in images]), 'only numpy array is supported'
            shapes = [list(i.shape) for i in images]
            for s in shapes:
                s[self.axis] = None
            assert all([s == shapes[0] for s in shapes]
                       ), 'shapes must be the same except the merge axis'
            return np.concatenate(images, axis=self.axis)
        else:
            raise Exception("obj is not a sequence (list, tuple, etc)")


class Split(object):
    """Split images into individual arraies
    """

    def __init__(self, *slices, **kwargs):
        assert isinstance(slices, collections.Sequence)
        slices_ = []
        for s in slices:
            if isinstance(s, collections.Sequence):
                slices_.append(slice(*s))
            else:
                slices_.append(s)
        assert all([isinstance(s, slice) for s in slices_]
                   ), 'slices must be consist of slice instances'
        self.slices = slices_
        self.axis = kwargs.get('axis', -1)

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            ret = []
            for s in self.slices:
                sl = [slice(None)] * image.ndim
                sl[self.axis] = s
                ret.append(image[sl])
            return ret
        else:
            raise Exception("obj is not an numpy array")


class SaveImageTiff(object):
    """Save the image to tiff file
    """

    def __init__(self, path, prefix=''):
        self.prefix = prefix
        self.path = path
        self.counter = 0

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            for i in range(image.shape[2]):
                im = Image.fromarray(image[:, :, i].astype('float32'))
                im.save(os.path.join(self.path, self.prefix+'b'+str(self.counter)+'_c'+str(i)+'.tif'))
                self.counter += 1
        else:
            raise Exception("obj is not an numpy array")
        return image


class ElasticTransform(object):
    """Apply elastic transformation on a numpy.ndarray (H x W x C)
    """

    def __init__(self, alpha=1000, sigma=40):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image):
        if isinstance(self.alpha, collections.Sequence):
            alpha = num_generator(self.alpha)
        else:
            alpha = self.alpha
        if isinstance(self.sigma, collections.Sequence):
            sigma = num_generator(self.sigma)
        else:
            sigma = self.sigma
        return elastic_transform(image, alpha=alpha, sigma=sigma)


class PoissonSubsampling(object):
    """Poisson subsampling on a numpy.ndarray (H x W x C)
    """

    def __init__(self, peak, random_state=np.random):
        self.peak = peak
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.peak, collections.Sequence):
            peak = num_generator(
                self.peak, random_state=self.random_state)
        else:
            peak = self.peak
        return poisson_downsampling(image, peak, random_state=self.random_state)


class AddGaussianNoise(object):
    """Add gaussian noise to a numpy.ndarray (H x W x C)
    """

    def __init__(self, mean, sigma, random_state=np.random):
        self.sigma = sigma
        self.mean = mean
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.sigma, collections.Sequence):
            sigma = num_generator(
                self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma
        if isinstance(self.mean, collections.Sequence):
            mean = num_generator(self.mean, random_state=self.random_state)
        else:
            mean = self.mean
        row, col, ch = image.shape
        gauss = self.random_state.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        return image + gauss


class AddSpeckleNoise(object):
    """Add speckle noise to a numpy.ndarray (H x W x C)
    """

    def __init__(self, mean, sigma, random_state=np.random):
        self.sigma = sigma
        self.mean = mean
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.sigma, collections.Sequence):
            sigma = num_generator(
                self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma
        if isinstance(self.mean, collections.Sequence):
            mean = num_generator(
                self.mean, random_state=self.random_state)
        else:
            mean = self.mean
        row, col, ch = image.shape
        gauss = self.random_state.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        return image * gauss


class GaussianBlurring(object):
    """Apply gaussian blur to a numpy.ndarray (H x W x C)
    """

    def __init__(self, sigma, random_state=np.random):
        self.sigma = sigma
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.sigma, collections.Sequence):
            sigma = num_generator(
                self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma
        image = gaussian_filter(image, sigma=(sigma, sigma, 0))
        return image


class AddGaussianPoissonNoise(object):
    """Add poisson noise with gaussian blurred image to a numpy.ndarray (H x W x C)
    """

    def __init__(self, sigma, peak, random_state=np.random):
        self.sigma = sigma
        self.peak = peak
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.sigma, collections.Sequence):
            sigma = num_generator(
                self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma
        if isinstance(self.peak, collections.Sequence):
            peak = num_generator(
                self.peak, random_state=self.random_state)
        else:
            peak = self.peak
        bg = gaussian_filter(image, sigma=(sigma, sigma, 0))
        bg = poisson_downsampling(
            bg, peak=peak, random_state=self.random_state)
        return image + bg


class MaxScaleNumpy(object):
    """scale with max and min of each channel of the numpy array
    """

    def __init__(self, range_min=0.0, range_max=1.0):
        self.scale = (range_min, range_max)

    def __call__(self, image):
        mn = image.min(axis=(0, 1))
        mx = image.max(axis=(0, 1))
        if np.all(mx-mn) == 0:
            return image
        return self.scale[0] + (image - mn) * (self.scale[1] - self.scale[0]) / (mx - mn)


class ScaleNumpy(object):
    """scale with the numpy array
    """
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, image):
        return image*self.scale


class ClipNumpy(object):
    """clip the array with max and min
    """

    def __init__(self, range_min=0.0, range_max=255.0):
        self.range_min = range_min
        self.range_max = range_max

    def __call__(self, image):
        return np.clip(image, self.range_min, self.range_max)


class MedianClipNumpy(object):
    """Clip with median of the numpy array
    """

    def __init__(self, range_min=0.0, median_factor_max=2.0):
        self.range_min = range_min
        self.median_factor_max = median_factor_max

    def __call__(self, image):
        md = np.median(image)
        return np.clip(image, self.range_min, md*self.median_factor_max)


class UniqueMedianClipNumpy(object):
    """Clip with median of the numpy array
    """

    def __init__(self, range_min=0.0, median_factor_max=2.0):
        self.range_min = range_min
        self.median_factor_max = median_factor_max

    def __call__(self, image):
        md = np.median(np.unique(image))
        return np.clip(image, self.range_min, md*self.median_factor_max)


class MedianScaleNumpy(object):
    """Scale with median and mean of each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, range_min=0.0, range_max=1.0):
        self.scale = (range_min, range_max)

    def __call__(self, image):
        mn = image.min(axis=(0, 1))
        md = np.median(image, axis=(0, 1))
        return self.scale[0] + (image - mn) * (self.scale[1] - self.scale[0]) / (md - mn)


class NormalizeNumpy(object):
    """Normalize each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """

    def __call__(self, image):
        image -= image.mean(axis=(0, 1))
        s = image.std(axis=(0, 1))
        s[s == 0] = 1.0
        image /= s
        return image


class MutualExclude(object):
    """Remove elements from one channel
    """

    def __init__(self, exclude_channel, from_channel):
        self.from_channel = from_channel
        self.exclude_channel = exclude_channel

    def __call__(self, image):
        mask = image[:, :, self.exclude_channel] > 0
        image[:, :, self.from_channel][mask] = 0
        return image


class RandomCropNumpy(object):
    """Crops the given numpy array at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, random_state=np.random):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = (int(size[0]), int(size[1]))
        self.random_state = random_state

    def __call__(self, img):
        h, w = img.shape[:2]
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = self.random_state.randint(0, w - tw)
        y1 = self.random_state.randint(0, h - th)
        return img[y1: y1 + th, x1:x1 + tw, :]


class CenterCropNumpy(object):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = (int(size[0]), int(size[1]))

    def __call__(self, img):
        h, w = img.shape[:2]
        th, tw = self.size
        assert th <= h and tw <= w, 'input size ({}, {}) is smaller than crop size({}, {}).'.format(h, w, th, tw)
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[y1: y1 + th, x1:x1 + tw,  :]


class RandomRotate(object):
    """Rotate a PIL.Image or numpy.ndarray (H x W x C) randomly
    """

    def __init__(self, angle_range=(0.0, 360.0), axes=(0, 1), mode='reflect', random_state=np.random):
        assert isinstance(angle_range, tuple)
        self.angle_range = angle_range
        self.random_state = random_state
        self.axes = axes
        self.mode = mode

    def __call__(self, image):
        angle = self.random_state.uniform(
            self.angle_range[0], self.angle_range[1])
        if isinstance(image, np.ndarray):
            mi, ma = image.min(), image.max()
            image = scipy.ndimage.interpolation.rotate(
                image, angle, reshape=False, axes=self.axes, mode=self.mode)
            return np.clip(image, mi, ma)
        elif isinstance(image, Image.Image):
            return image.rotate(angle)
        else:
            raise Exception('unsupported type')


class BilinearResize(object):
    """Resize a PIL.Image or numpy.ndarray (H x W x C)
    """

    def __init__(self, zoom):
        self.zoom = [zoom, zoom, 1]

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            return scipy.ndimage.interpolation.zoom(image, self.zoom)
        elif isinstance(image, Image.Image):
            return image.resize(self.size, Image.BILINEAR)
        else:
            raise Exception('unsupported type')

class DictCopy(object):
    """Copy items in the dict
    """

    def __init__(self, map):
        assert type(map) is dict
        self.map = map

    def __call__(self, dic):
        return {k: dic[v] for k, v in self.map.items()}


class Dict2List(object):
    """Convert a dictionary to a list
    """

    def __init__(self, keys):
        assert type(keys) is list
        self.keys = keys

    def __call__(self, dic):
        return [dic[k] for k in self.keys]

class GroupList(object):
    """Convert a dictionary to a list
    """

    def __init__(self, groups):
        assert type(groups) is list
        self.groups = groups

    def __call__(self, li):
        assert type(li) is list
        ret = []
        for g in self.groups:
            if type(g) is list:
                ret.append([li[v] for v in g])
            elif type(g) is int:
                ret.append(li[g])
            else:
                raise Exception('not supported group type')
        return ret

class UngroupList(object):
    """Convert a dictionary to a list
    """
    def __call__(self, li):
        assert type(li) is list
        ret = []
        for g in li:
            if type(g) is list:
                for v in g:
                    ret.append(v)
            else:
                ret.append(g)

        return ret

class List2Dict(object):
    """Convert a list to a dictionary
    """

    def __init__(self, keys):
        assert type(keys) is list
        self.keys = keys

    def __call__(self, li):
        return {k: li[i] for i, k in enumerate(self.keys)}


class EnhancedCompose(object):
    """Composes several transforms together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, collections.Sequence):
                assert isinstance(img, collections.Sequence) and len(img) == len(
                    t), "size of image group and transform group does not fit"
                tmp_ = []
                for i, im_ in enumerate(img):
                    if callable(t[i]):
                        tmp_.append(t[i](im_))
                    else:
                        tmp_.append(im_)
                img = tmp_
            elif type(t) is dict:
                assert type(img) is dict, 'input data must be dictionary'
                tmp_ = {}
                for k in img:
                    if k in t and callable(t[k]):
                        tmp_[k] = t[k](img[k])
                    else:
                        tmp_[k] = img[k]
                img = tmp_
            elif callable(t):
                img = t(img)
            elif t is None:
                continue
            else:
                raise Exception('unexpected transform type:' + str(t))
        return img


class FolderImagesLoader(FileLoader):
    def __init__(self, extension='png'):
        self.__cache = {}
        self.ext = extension

    def load(self, path):
        if path not in self.__cache:
            self.cache(path)
        return self.__cache[path]

    def cache(self, path):
        self.__cache[path] = np.array(Image.open(path))

    def __call__(self, path):
        if path not in self.__cache:
            self.cache(path)
        return self.__cache[path]


if __name__ == '__main__':
    input_channel = 3
    target_channel = 3
    transform = EnhancedCompose([
        Merge(),
        RandomCropNumpy(size=(512, 512)),
        RandomRotate(),
        Split([0, input_channel], [input_channel, input_channel+target_channel]),
        [CenterCropNumpy(size=(256, 256)), CenterCropNumpy(size=(256, 256))],
        [NormalizeNumpy(), MaxScaleNumpy(0, 1.0)],
    ])
    image_in = np.array(Image.open('input.jpg'))
    image_target = np.array(Image.open('target.jpg'))
    x, y = transform([image_in, image_target])

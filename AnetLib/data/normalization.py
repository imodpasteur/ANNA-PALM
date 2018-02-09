import numpy as np

class NormalizeNumpy(object):
    """Normalize each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean=None, std=None, epsilon=1e-12):
        self.epsilon = epsilon
        self.mean = mean
        self.std = std

    def __call__(self, input_img):
        if self.mean is not None:
            mean = self.mean
        else:
            mean = input_img.mean(axis=(0, 1))
        if self.std is not None:
            std = self.std
        else:
            std = input_img.std(axis=(0, 1))
        image = input_img
        image = image - mean
        empty = std == 0
        std[std < self.epsilon] = self.epsilon
        image = image / std
        # copy empty image
        image[:, :, empty] =  input_img[:, :, empty]
        return image

class MaxScaleNumpy(object):
    """scale with max and min of each channel of the numpy array
    """

    def __init__(self, range_min=0.0, range_max=1.0, min_clip=[0, 254], max_clip=[1, 255]):
        self.scale = (range_min, range_max)
        self.min_clip = min_clip
        self.max_clip = max_clip

    def __call__(self, image):
        mn = np.clip(image.min(axis=(0, 1)), self.min_clip[0], self.min_clip[1])
        mx = np.clip(image.max(axis=(0, 1)), self.max_clip[0], self.max_clip[1])
        empty = mx-mn == 0
        out = self.scale[0] + (image - mn) * (self.scale[1] - self.scale[0]) / (mx - mn)
        out[:, :, empty] = image[:, :, empty]
        return out

def anscombe_transform(self, x):
        return 2*np.sqrt(x+3.0/8.0)

class AnscombeTransform(object):
    def __call__(self, image):
        s = image.std(axis=(0, 1))
        empty = s == 0
        out = 2*np.sqrt(image+3.0/8.0)
        out[:, :, empty] =  image[:, :, empty]
        return out

def get_norm(name):
    if name is None:
        return None
    name = name.strip()
    if '[' in name and name.endswith(']'):
        i = name.index('[')
        name, args = name[:i], name[i+1:-1]
        args =[p.strip() for p in args.split(',')]
        for i, a in enumerate(args):
            if a.isnumeric():
                if '.' in a:
                    args[i] = float(a)
                else:
                    args[i] = int(a)
    else:
        args = []

    if name == 'mean_std':
        return NormalizeNumpy()
    elif name == 'mean_std' and len(args) == 2:
        return NormalizeNumpy(args[0], args[1])
    elif name == 'clip' and len(args) == 2:
        return ClipNumpy(args[0], args[1])
    elif name == 'min_max' and len(args) == 2:
        return MaxScaleNumpy(args[0], args[1], min_clip=[0, 254], max_clip=[1, 255])
    elif name == 'anscombe':
        return AnscombeTransform()
    elif name == 'none':
        return lambda x: x
    else:
        raise NotImplemented

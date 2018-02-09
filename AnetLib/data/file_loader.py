import os
import numpy as np
from PIL import Image


class FileLoader():
    def cache(self, path):
        return True
    def save_cache(self, cache_path):
        pass
    def load_cache(self, cache_path):
        pass
    def __call__(self, path):
        raise NotImplemented


class ImageLoader(FileLoader):
    def __init__(self, mode="F"):
        self.mode = mode
    def cache(self, path):
        return True
    def __call__(self, path):
        return Image.open(path).convert(self.mode)


class ImageJRoi2Mask(FileLoader):
    def __init__(self, image_size):
        if type(image_size) is int:
            image_size = (image_size, image_size)
        assert len(image_size) == 2
        self.image_size = image_size

        self.__cache = {}

    def generate(self, roi_list):
        from skimage import draw
        image_size = self.image_size
        mask_fill = np.zeros(image_size + (1, ), dtype=np.uint8)
        rr_all = []
        cc_all = []
        for i, roi in enumerate(roi_list):
            # Draw polygon and add it to image
            rr, cc = draw.polygon(roi[:, 0], roi[:, 1])
            rr[rr < 0] = 0
            rr[rr > image_size[0] - 1] = image_size[0] - 1
            cc[cc < 0] = 0
            cc[cc > image_size[0] - 1] = image_size[0] - 1
            # test if this region has already been added
            if any(np.array_equal(rr, rr_test) for rr_test in rr_all) and any(np.array_equal(cc, cc_test) for cc_test in cc_all):
                # print('Region #{} has already been used'.format(i + 1))
                continue
            rr_all.append(rr)
            cc_all.append(cc)
            # Generate mask
            mask_fill[rr, cc, :] = 1
        return mask_fill

    def cache(self, path):
        self.__cache[path] = self.generate(read_roi_zip(path))
        return True

    def __call__(self, path):
        return self.__cache[path]

class ImageJRoi2Edge(ImageJRoi2Mask):
    def __init__(self, image_size, erose_size=5):
        super(ImageJRoi2Edge, self).__init__(image_size)
        self.erose_size = erose_size

    def generate(self, roi_list):
        from skimage import morphology, draw
        image_size = self.image_size
        mask_edge = np.zeros(image_size + (1, ), dtype=np.uint8)
        rr_all = []
        cc_all = []
        for i, roi in enumerate(roi_list):
            # Draw polygon and add it to image
            rr, cc = draw.polygon(roi[:, 0], roi[:, 1])
            rr[rr < 0] = 0
            rr[rr > image_size[0] - 1] = image_size[0] - 1
            cc[cc < 0] = 0
            cc[cc > image_size[0] - 1] = image_size[0] - 1
            # test if this region has already been added
            if any(np.array_equal(rr, rr_test) for rr_test in rr_all) and any(np.array_equal(cc, cc_test) for cc_test in cc_all):
                # print('Region #{} has already been used'.format(i + 1))
                continue
            rr_all.append(rr)
            cc_all.append(cc)

            # Generate mask
            mask_fill_roi = np.zeros(image_size, dtype=np.uint8)
            mask_fill_roi[rr, cc] = 1

            # Erode to get cell edge - both arrays are boolean to be used as
            # index arrays later
            mask_fill_roi_erode = morphology.binary_erosion(
                mask_fill_roi, np.ones((self.erose_size, self.erose_size)))
            mask_edge_roi = (mask_fill_roi.astype('int') - mask_fill_roi_erode.astype('int')).astype('bool')
            mask_edge[mask_edge_roi] = 1

        return mask_edge


class ImageJRoi2DistanceMap(ImageJRoi2Mask):
    def __init__(self, image_size, truncate_distance=None):
        super(ImageJRoi2DistanceMap, self).__init__(image_size)
        self.truncate_distance = truncate_distance

    def generate(self, roi_list):
        from scipy import ndimage
        mask = super(ImageJRoi2DistanceMap, self).generate(roi_list)
        dist = ndimage.distance_transform_edt(mask)
        if self.truncate_distance:
            dist[dist > self.truncate_distance] = self.truncate_distance
        return dist


'''
    ====== CODE TO READ FIJI ROI FILES
    # read_roi  & read_roi_zip
    Copyright: Luis Pedro Coelho <luis@luispedro.org>, 2012
    # License: MIT

    Smalle changes to adapt for Python 3 (Florian Mueller)
'''


def read_roi(fileobj):
    '''
    points = read_roi(fileobj)

    Read ImageJ's ROI format
    '''
    # This is based on:
    # http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html

    SPLINE_FIT = 1
    DOUBLE_HEADED = 2
    OUTLINE = 4
    OVERLAY_LABELS = 8
    OVERLAY_NAMES = 16
    OVERLAY_BACKGROUNDS = 32
    OVERLAY_BOLD = 64
    SUB_PIXEL_RESOLUTION = 128
    DRAW_OFFSET = 256

    pos = [4]

    def get8():
        pos[0] += 1
        s = fileobj.read(1)
        if not s:
            raise IOError('readroi: Unexpected EOF')
        return ord(s)

    def get16():
        b0 = get8()
        b1 = get8()
        return (b0 << 8) | b1

    def get32():
        s0 = get16()
        s1 = get16()
        return (s0 << 16) | s1

    def getfloat():
        v = np.int32(get32())
        return v.view(np.float32)

    magic = fileobj.read(4)
    if magic != b'Iout':
        raise IOError('Magic number not found')
    version = get16()

    # It seems that the roi type field occupies 2 Bytes, but only one is used
    annot_type = get8()
    # Discard second Byte:
    get8()

    if not (0 <= annot_type < 11):
        raise ValueError('roireader: ROI type %s not supported' % annot_type)

    if annot_type != 7:
        raise ValueError(
            'roireader: ROI type %s not supported (!= 7)' % annot_type)

    top = get16()
    left = get16()
    bottom = get16()
    right = get16()
    n_coordinates = get16()

    x1 = getfloat()
    y1 = getfloat()
    x2 = getfloat()
    y2 = getfloat()
    stroke_width = get16()
    shape_roi_size = get32()
    stroke_color = get32()
    fill_color = get32()
    subtype = get16()
    if subtype != 0:
        raise ValueError(
            'roireader: ROI subtype %s not supported (!= 0)' % subtype)
    options = get16()
    arrow_style = get8()
    arrow_head_size = get8()
    rect_arc_size = get16()
    position = get32()
    header2offset = get32()

    if options & SUB_PIXEL_RESOLUTION:
        getc = getfloat
        points = np.empty((n_coordinates, 2), dtype=np.float32)
    else:
        getc = get16
        points = np.empty((n_coordinates, 2), dtype=np.int16)
    points[:, 1] = [getc() for i in range(n_coordinates)]
    points[:, 0] = [getc() for i in range(n_coordinates)]
    points[:, 1] += left
    points[:, 0] += top
    points -= 1
    return points


def read_roi_zip(fname):
    if not os.path.exists(fname):
        print('zip file not found: '+ fname)
        return []
    import zipfile
    with zipfile.ZipFile(fname) as zf:
        return [read_roi(zf.open(n))
                for n in zf.namelist()]

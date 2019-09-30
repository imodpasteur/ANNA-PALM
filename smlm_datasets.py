import random
import os
import numpy as np
import scipy
from PIL import Image

from AnetLib.data.image_utils import RandomRotate, CenterCropNumpy, RandomCropNumpy, PoissonSubsampling, AddGaussianPoissonNoise, GaussianBlurring, AddGaussianNoise, ElasticTransform
from datasets import TUBULIN, NUCLEAR_PORE
from AnetLib.data.file_loader import FileLoader,ImageLoader
from AnetLib.data.image_utils import EnhancedCompose, Merge, Split
from AnetLib.data.image_utils import RandomRotate, CenterCropNumpy, RandomCropNumpy
from AnetLib.data.image_utils import NormalizeNumpy, MaxScaleNumpy
from AnetLib.data.folder_dataset import FolderDataset, SubfolderDataset
from localization_utils import generate_image_pairs_from_csv, SubFolderImagesLoader

DatasetTypeIDs = {'random': -1, 'microtubule': 0, 'nuclear_pore': 1, 'actin': 2, 'mitochondria': 3}

def create_data_sources(name, opt):
    np.random.seed(opt.seed)
    if type(name) is dict or type(name) is list:
        return CompositeDataset(name, opt)
    if name == 'TransformedTubulin001':
        return TransformedTubulin001(opt)
    elif name == 'TransformedTubulin001NB':
        # no gaussian blur
        return TransformedTubulin001NB(opt)
    elif name == 'TransformedTubulin001DenseNB':
        # no gaussian blur, dense input
        return TransformedTubulin001DenseNB(opt)
    elif name == 'TransformedTubulin002':
        return TransformedTubulin002(opt)
    elif name == 'TransformedTubulin003':
        return TransformedTubulin003(opt)
    elif name == 'TransformedTubulin004':
        return TransformedTubulin004(opt)
    elif name == 'TransformedTubulin005':
        return TransformedTubulin005(opt)
    elif name == 'TransformedLRSR':
        return TransformedLRSR(opt)
    elif name == 'TransformedLRSR002':
        return TransformedLRSR002(opt)
    elif name == 'TransformedNuclearPore001':
        return TransformedNuclearPore001(opt)
    elif name == 'TransformedNuclearPore001Dense':
        return TransformedNuclearPore001Dense(opt)
    elif name == 'TransformedCSVImages':
        return TransformedCSVImages(opt)
    elif name == 'TransformedABImages':
        return TransformedABImages(opt)
    elif name == 'GenericTransformedImages':
        return GenericTransformedImages(opt)
    else:
        raise Exception('unsupported dataset')


class TransformedTubulin001():
    def __init__(self, opt):
        self.typeID = DatasetTypeIDs['microtubule']
        self.tags = ['microtubule', 'simulation']
        self.iRot = RandomRotate()
        self.iMerge = Merge()
        self.iSplit = Split([0, 1], [1, 2])
        self.irCropTrain = RandomCropNumpy(size=(opt.fineSize+100, opt.fineSize+100))
        self.ioCropTrain = CenterCropNumpy(size=[opt.fineSize, opt.fineSize])
        self.iCropTest = CenterCropNumpy(size=(1024, 1024))
        self.iElastic = ElasticTransform(alpha=1000, sigma=40)
        self.iBlur = GaussianBlurring(sigma=1.5)
        self.iPoisson = PoissonSubsampling(peak=['lognormal', -0.5, 0.001])
        self.iBG = AddGaussianPoissonNoise(sigma=25, peak=0.06)
        self.train_count = 0
        self.test_count = 0
        self.dim_ordering = opt.dim_ordering
        self.repeat = 1
        self.opt = opt

    def __getitem__(self, key):
        if key == 'train':
            source_train = TUBULIN('./datasets', train=True, download=True, transform=self.transform_train, repeat=self.repeat)
            return source_train
        elif key == 'test':
            source_test = TUBULIN('./datasets', train=False, download=True, transform=self.transform_test, repeat=self.repeat)
            return source_test
        else:
            raise Exception('only train and test are supported.')

    def transform_train(self, imageIO):
        img = self.iMerge(imageIO.copy())
        img = self.irCropTrain(img)
        img = self.iRot(img)
        img = self.ioCropTrain(img)
        img = self.iElastic(img)
        iIm, oIm = self.iSplit(img)
        iIm, oIm = self.iBlur(iIm), self.iBlur(oIm)

        imgin, imgout = self.iBG(self.iPoisson(iIm)), oIm
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'

        path = str(self.train_count)
        self.train_count += 1
        return {'A': imgin, 'B': imgout, 'path': path}

    def transform_test(self, imageIO):
        img = self.iMerge(imageIO.copy())
        img = self.irCropTrain(img)
        # img = iRot(img)
        img = self.ioCropTrain(img)
        iIm, oIm = self.iSplit(img)
        iIm, oIm = self.iBlur(iIm), self.iBlur(oIm)
        imgin, imgout = self.iBG(self.iPoisson(iIm)), oIm
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        path = str(self.test_count)
        self.test_count += 1
        return {'A': imgin, 'B': imgout, 'path': path}

class TransformedTubulin001NB(TransformedTubulin001):
    def __init__(self, opt):
        super(TransformedTubulin001NB, self).__init__(opt)
        self.iBlur = lambda x: x #GaussianBlurring(sigma=1.5)

class TransformedTubulin001DenseNB(TransformedTubulin001):
    def __init__(self, opt):
        super(TransformedTubulin001DenseNB, self).__init__(opt)
        self.iBlur = lambda x: x #GaussianBlurring(sigma=1.5)
        self.iPoisson = PoissonSubsampling(peak=['lognormal', 0.8, 1])

class TransformedNuclearPore001(TransformedTubulin001):
    def __init__(self, opt):
        super(TransformedNuclearPore001, self).__init__(opt)
        self.typeID = DatasetTypeIDs['nuclear_pore']
        self.tags = ['nuclear_pore', 'simulation']
    def __getitem__(self, key):
        if key == 'train':
            source_train = NUCLEAR_PORE('./datasets', train=True, download=True, transform=self.transform_train, repeat=self.repeat)
            return source_train
        elif key == 'test':
            source_test = NUCLEAR_PORE('./datasets', train=False, download=True, transform=self.transform_test, repeat=self.repeat)
            return source_test
        else:
            raise Exception('only train and test are supported.')

class TransformedNuclearPore001Dense(TransformedNuclearPore001):
    def __init__(self, opt):
        super(TransformedNuclearPore001Dense, self).__init__(opt)
        self.iPoisson = PoissonSubsampling(peak=['lognormal', 0.8, 1])

class TransformedTubulinImages001():
    def __init__(self, opt):
        self.typeID = DatasetTypeIDs['microtubule']
        train_crop_size1 = opt.fineSize * 2
        train_crop_size2 = opt.fineSize + 200
        train_crop_size3 = opt.fineSize
        test_size = opt.fineSize

        self.input_clip = (0, 5)
        self.output_clip = (2, 100)

        # prepare the transforms
        self.iMerge = Merge()
        self.iElastic = ElasticTransform(alpha=1000, sigma=40)
        self.iSplit = Split([0, 1], [1, 2])
        self.iRot = RandomRotate()
        self.iRCropTrain = RandomCropNumpy(size=(train_crop_size2, train_crop_size2))
        self.iCropFTrain = CenterCropNumpy(size=(train_crop_size1, train_crop_size1))
        self.iCropTrain = CenterCropNumpy(size=(train_crop_size3, train_crop_size3))
        self.iCropTest = CenterCropNumpy(size=(test_size, test_size))
        self.ptrain = './datasets/wei-tubulin-ctrl-20170520-images/train'
        self.ptest = './datasets/wei-tubulin-ctrl-20170520-images/test'
        self.dim_ordering = opt.dim_ordering
        self.opt = opt
        self.repeat = 30
        self.folder_filter = '*.csv'
        self.file_extension = '.png'

    def __getitem__(self, key):
        if key == 'train':
            imgfolderLoader = SubFolderImagesLoader(extension=self.file_extension)
            source_train = FolderDataset(self.ptrain,
                              channels = {'image': {'filter': self.folder_filter, 'loader': imgfolderLoader} },
                             transform = self.transform_train,
                             recursive=False,
                             repeat=self.repeat)
            return source_train
        elif key == 'test':
            imgfolderLoader = SubFolderImagesLoader(extension=self.file_extension)
            source_test = FolderDataset(self.ptest,
                              channels = {'image': {'filter': self.folder_filter, 'loader': imgfolderLoader} },
                             transform = self.transform_test,
                             recursive=False,
                             repeat=self.repeat)
            return source_test
        else:
            raise Exception('only train and test are supported.')

    def transform_train(self, imageAB):
        As, Bs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image.path']
        histin, histout = random.choice(As).astype('float32'), random.choice(Bs).astype('float32')
        histin = np.expand_dims(histin, axis=2) if histin.ndim == 2 else histin
        histout = np.expand_dims(histout, axis=2) if histout.ndim == 2 else histout
        img = self.iMerge([histin, histout])
        img = self.iRCropTrain(img)
        img = self.iRot(img)
        img = self.iElastic(img)
        histin, histout = self.iSplit(img)
        output_clip = self.output_clip
        histout =  (np.clip(histout, output_clip[0]/2, output_clip[1]*2)-output_clip[0]) / (output_clip[1] - output_clip[0])
        imgin, imgout = self.iCropTrain(histin), self.iCropTrain(histout)
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}

    def transform_test(self, imageAB):
        As, Bs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image.path']
        histin, histout = random.choice(As).astype('float32'), random.choice(Bs if len(Bs)>0 else As).astype('float32')
        histin = np.expand_dims(histin, axis=2) if histin.ndim == 2 else histin
        histout = np.expand_dims(histout, axis=2) if histout.ndim == 2 else histout
        output_clip = self.output_clip
        histout =  (np.clip(histout, output_clip[0]/2, output_clip[1]*2)-output_clip[0]) / (output_clip[1] - output_clip[0])
        imgin, imgout = self.iCropTest(histin), self.iCropTest(histout)
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}

class TransformedTubulinImages004(TransformedTubulinImages001):
    '''
    use max scale and normalization
    filter out blank regions
    '''
    def __init__(self, opt):
        super(TransformedTubulinImages004, self).__init__(opt)
        self.iBG = lambda x: x # AddGaussianPoissonNoise(sigma=25, peak=['lognormal', -2.5, 0.8])

    def transform_train(self, imageAB):
        As, Bs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image.path']
        histin, histout = random.choice(As).astype('float32'), random.choice(Bs).astype('float32')
        histin = np.expand_dims(histin, axis=2) if histin.ndim == 2 else histin
        histout = np.expand_dims(histout, axis=2) if histout.ndim == 2 else histout
        img = self.iMerge([histin, histout])

        # find a non-black region
        retry = 0
        while retry<5:
            img_crop = self.iRCropTrain(img)
            if img_crop[:, :, 0].sum()>800+np.random.random()*800:
                break
            retry +=1
        if retry>=5:
            print('X', end='')

        img = img_crop
        img = self.iRot(img)
        img = self.iElastic(img)
        histin, histout = self.iSplit(img)
        histin, histout = self.iBG(histin), histout
        imgin, imgout = self.iCropTrain(histin), self.iCropTrain(histout)
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}

    def transform_test(self, imageAB):
        As, Bs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image.path']
        histin, histout = random.choice(As).astype('float32'), random.choice(Bs if len(Bs)>0 else As).astype('float32')
        histin = np.expand_dims(histin, axis=2) if histin.ndim == 2 else histin
        histout = np.expand_dims(histout, axis=2) if histout.ndim == 2 else histout
        imgin, imgout = self.iCropTest(histin), self.iCropTest(histout)
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}

class NoiseCollection001(TransformedTubulinImages004):
    def __init__(self, opt, force_generate=False):
        super(NoiseCollection001, self).__init__(opt)
        self.typeID = DatasetTypeIDs['random']
        self.ptrain = './datasets/noise-collection_v0.1.0/train'
        self.ptest = './datasets/noise-collection_v0.1.0/test'

    def transform_train(self, imageAB):
        As, Bs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image.path']
        histin, histout = random.choice(As).astype('float32'), random.choice(Bs).astype('float32')
        histin = np.expand_dims(histin, axis=2) if histin.ndim == 2 else histin
        histout = np.expand_dims(histout, axis=2) if histout.ndim == 2 else histout
        img = self.iMerge([histin, histout])
        img_crop = self.iRCropTrain(img)
        img = img_crop
        img = self.iRot(img)
        img = self.iElastic(img)
        img = self.iCropTrain(img)
        imgin, imgout = self.iSplit(img)
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}

class TransformedCSVImages(TransformedTubulinImages004):
    def __init__(self, opt, force_generate=False):
        super(TransformedCSVImages, self).__init__(opt)
        self.ptrain = os.path.join(opt.workdir, '__images__', 'train')
        self.ptest = os.path.join(opt.workdir, '__images__', 'test')
        self.iSplit = Split([0, 2], [2, 3])
        self.test_count = 0
        if not os.path.exists(self.ptrain) or force_generate:
            generate_image_pairs_from_csv(os.path.join(opt.workdir, 'train'),
                                    self.ptrain,
                                    A_frame=['uniform', 200, 500], B_frame=0.95,
                                    A_frame_limit=(0, 0.5),
                                    B_frame_limit=(2000, 1.0),
                                    image_per_file=30,
                                    target_size=(2560, 2560))

        if not os.path.exists(self.ptest) or force_generate:
            if not os.path.exists(os.path.join(opt.workdir, 'test')):
                return
            aframes = list(np.logspace(-3, np.log(1.0), 32)*60000) + [0,]
            generate_image_pairs_from_csv(os.path.join(opt.workdir, 'test'),
                                    self.ptest,
                                    A_frame=aframes, B_frame=1.0,
                                    A_frame_limit=(0, 1.0),
                                    B_frame_limit=(0, 1.0),
                                    image_per_file=len(aframes),
                                    target_size=(2560, 2560),
                                    zero_offset=True)

    def transform_train(self, imageAB):
        As, Bs, LRs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image']['LR'], imageAB['image.path']
        histin, histout = random.choice(As).astype('float32'), random.choice(Bs).astype('float32')
        if len(LRs) == 0:
            lrin = np.zeros_like(histin)
        else:
            lrin = random.choice(LRs).astype('float32')
        img = self.iMerge([histin, lrin, histout])
        img = self.iRCropTrain(img)
        img = self.iRot(img)
        img = self.iElastic(img)
        histin, histout = self.iSplit(img)
        output_clip = self.output_clip
        histout =  (np.clip(histout, output_clip[0]/2, output_clip[1]*2)-output_clip[0]) / (output_clip[1] - output_clip[0])
        imgin, imgout = self.iCropTrain(histin), self.iCropTrain(histout)
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}

    def transform_test(self, imageAB):
        As, Bs, LRs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image']['LR'], imageAB['image.path']
        pathAs = imageAB['image']['pathA']
        if self.test_count >= len(As):
            self.test_count = 0

        path = path + '_' + os.path.split(pathAs[self.test_count%len(As)])[1]
        histin, histout= As[self.test_count%len(As)].astype('float32'), (Bs if len(Bs)>0 else As)[self.test_count%len(Bs if len(Bs)>0 else As)].astype('float32')
        self.test_count += 1
        if len(LRs) == 0:
            lrin = np.zeros_like(histin)
        else:
            lrin = random.choice(LRs).astype('float32')
        histin = np.concatenate([histin, lrin], axis=2)
        output_clip = self.output_clip
        histout =  (np.clip(histout, output_clip[0]/2, output_clip[1]*2)-output_clip[0]) / (output_clip[1] - output_clip[0])
        imgin, imgout = self.iCropTest(histin), self.iCropTest(histout)
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}


class TransformedABImages(TransformedCSVImages):
    def __init__(self, opt, force_generate=False):
        super(TransformedCSVImages, self).__init__(opt)
        self.ptrain = os.path.join(opt.workdir, 'train')
        self.ptest = os.path.join(opt.workdir, 'test')
        self.iSplit = Split([0, 2], [2, 3])
        self.test_count = 0
        self.folder_filter = '*'
        self.file_extension = '.png'

class SubFolderWFImagesLoader(FileLoader):
    def __init__(self, drift_correction=False, scale_LR=True):
        self.__cache = {}
        self.ext = 'tif'
        self.drift_correction = drift_correction
        self.scale_LR = scale_LR

    def load(self, path):
        if path not in self.__cache:
            self.cache(path)
        return self.__cache[path]

    def cache(self, path):
        Bs = [os.path.join(path, p) for p in os.listdir(path) if p == 'Histograms.tif']
        LRs = [os.path.join(path, p) for p in os.listdir(path) if p == 'WF_TMR_calibrated.tif']
        ImgBs, PathBs, ImgLRs, PathLRs= [], [], [], []
        for p in Bs:
            img = np.array(Image.open(p))
            img = np.expand_dims(img, axis=2) if img.ndim == 2 else img
            ImgBs.append(img)
            PathBs.append(p)

        for p in LRs:
            try:
                imgStack = Image.open(p)
                indexes = [i for i in range(imgStack.n_frames)]
                random.shuffle(indexes)
                c = min(len(indexes), 20)
                for i in indexes[:c]:
                    imgStack.seek(i)
                    img = np.array(imgStack)
                    dtype = img.dtype
                    assert img.ndim == 2
                    if self.drift_correction:
                        import imreg_dft as ird
                        from skimage import exposure
                        b = ImgBs[0][:, :, 0]
                        b = exposure.equalize_hist(b)
                        b = scipy.ndimage.filters.gaussian_filter(b, sigma=(6, 6))
                        b = scipy.misc.imresize(b, img.shape[:2])
                        ts = ird.translation(b, img)
                        tvec = ts["tvec"]
                        # the Transformed IMaGe.
                        img = ird.transform_img(img, tvec=tvec)
                    if self.scale_LR == True:
                        img = scipy.misc.imresize(img, ImgBs[0].shape[:2])
                    elif type(self.scale_LR) is list:
                        img = scipy.misc.imresize(img, self.scale_LR)
                    img = np.expand_dims(img, axis=2)
                    img = img.astype(dtype)
                    ImgLRs.append(img)
                    PathLRs.append(p)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print('error when reading file ', p)
                import traceback, sys
                traceback.print_exc(file=sys.stdout)

        self.__cache[path] = { 'B': ImgBs, 'A':ImgLRs, 'path': path, 'pathB': PathBs, 'pathA': PathLRs}
        return True

    def __call__(self, path):
        if path not in self.__cache:
            self.cache(path)
        return self.__cache[path].copy()


class TransformedLRSR():
    def __init__(self, opt):
        self.typeID = DatasetTypeIDs['microtubule']
        train_crop_size1 = opt.fineSize * 2
        train_crop_size2 = opt.fineSize + 200
        train_crop_size3 = opt.fineSize
        test_size = opt.fineSize

        self.input_clip = (0, 5)
        self.output_clip = (2, 100)

        # prepare the transforms
        self.iMerge = Merge()
        self.iElastic = ElasticTransform(alpha=1000, sigma=40)
        self.iSplit = Split([0, 1], [1, 2])
        self.iRot = RandomRotate()
        self.iRCropTrain = RandomCropNumpy(size=(train_crop_size2, train_crop_size2))
        self.iCropFTrain = CenterCropNumpy(size=(train_crop_size1, train_crop_size1))
        self.iCropTrain = CenterCropNumpy(size=(train_crop_size3, train_crop_size3))
        self.iCropTest = CenterCropNumpy(size=(test_size, test_size))
        self.ptrain = '../anet-lite/src/datasets/Christian-TMR-IF-v0.1/train'
        self.ptest = '../anet-lite/src/datasets/Christian-TMR-IF-v0.1/test'
        self.dim_ordering = opt.dim_ordering
        self.opt = opt
        self.repeat = 30
        self.folder_filter = '*'
        self.drift_correction = False
        self.scale_LR = True

    def __getitem__(self, key):
        if key == 'train':
            imgfolderLoader = SubFolderWFImagesLoader(drift_correction=self.drift_correction, scale_LR=self.scale_LR)
            source_train = FolderDataset(self.ptrain,
                              channels = {'image': {'filter': self.folder_filter, 'loader': imgfolderLoader} },
                             transform = self.transform_train,
                             recursive=False,
                             repeat=self.repeat)
            return source_train
        elif key == 'test':
            imgfolderLoader = SubFolderWFImagesLoader(drift_correction=self.drift_correction, scale_LR=self.scale_LR)
            source_test = FolderDataset(self.ptest,
                              channels = {'image': {'filter': self.folder_filter, 'loader': imgfolderLoader} },
                             transform = self.transform_test,
                             recursive=False,
                             repeat=self.repeat)
            return source_test
        else:
            raise Exception('only train and test are supported.')

    def transform_train(self, imageAB):
        As, Bs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image.path']
        histin, histout = random.choice(As).astype('float32'), random.choice(Bs).astype('float32')
        histin = np.expand_dims(histin, axis=2) if histin.ndim == 2 else histin
        histout = np.expand_dims(histout, axis=2) if histout.ndim == 2 else histout
        img = self.iMerge([histin, histout])
        img = self.iRCropTrain(img)
        img = self.iRot(img)
        img = self.iElastic(img)
        histin, histout = self.iSplit(img)
        output_clip = self.output_clip
        histout =  (np.clip(histout, output_clip[0]/2, output_clip[1]*2)-output_clip[0]) / (output_clip[1] - output_clip[0])
        imgin, imgout = self.iCropTrain(histin), self.iCropTrain(histout)
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}

    def transform_test(self, imageAB):
        As, Bs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image.path']
        histin, histout = random.choice(As).astype('float32'), random.choice(Bs if len(Bs)>0 else As).astype('float32')
        histin = np.expand_dims(histin, axis=2) if histin.ndim == 2 else histin
        histout = np.expand_dims(histout, axis=2) if histout.ndim == 2 else histout
        output_clip = self.output_clip
        histout =  (np.clip(histout, output_clip[0]/2, output_clip[1]*2)-output_clip[0]) / (output_clip[1] - output_clip[0])
        imgin, imgout = self.iCropTest(histin), self.iCropTest(histout)
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}


class TransformedLRSR002(TransformedLRSR):
    def transform_train(self, imageAB):
        As, Bs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image.path']
        histin, histout = random.choice(As).astype('float32'), random.choice(Bs).astype('float32')
        histin = np.expand_dims(histin, axis=2) if histin.ndim == 2 else histin
        histout = np.expand_dims(histout, axis=2) if histout.ndim == 2 else histout
        img = self.iMerge([histin, histout])
        img = self.iRCropTrain(img)
        img = self.iRot(img)
        img = self.iElastic(img)
        histin, histout = self.iSplit(img)
        output_clip = self.output_clip
        histout =  (np.clip(histout, output_clip[0]/2, output_clip[1]*2)-output_clip[0]) / (output_clip[1] - output_clip[0])
        imgin, imgout = self.iCropTrain(histin), self.iCropTrain(histout)
        imgin = scipy.misc.imresize(imgin[:, :, 0], (self.opt.fineSize//4, self.opt.fineSize//4))[:, :, None]
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}

    def transform_test(self, imageAB):
        As, Bs, path = imageAB['image']['A'], imageAB['image']['B'], imageAB['image.path']
        histin, histout = random.choice(As).astype('float32'), random.choice(Bs if len(Bs)>0 else As).astype('float32')
        histin = np.expand_dims(histin, axis=2) if histin.ndim == 2 else histin
        histout = np.expand_dims(histout, axis=2) if histout.ndim == 2 else histout
        output_clip = self.output_clip
        histout =  (np.clip(histout, output_clip[0]/2, output_clip[1]*2)-output_clip[0]) / (output_clip[1] - output_clip[0])
        imgin, imgout = self.iCropTest(histin), self.iCropTest(histout)
        imgin = scipy.misc.imresize(imgin[:, :, 0], (self.opt.fineSize//4, self.opt.fineSize//4))[:, :, None]
        if self.dim_ordering == 'channels_first':
            imgin, imgout = imgin.transpose((2, 0, 1)), imgout.transpose((2, 0, 1))
        else:
            assert self.dim_ordering == 'channels_last'
        return {'A': imgin, 'B': imgout, 'path': path}



class GenericTransformedImages():
    def __init__(self, opt):
        train_crop_size1 = int(opt.fineSize * 1.45) #pre-crop
        train_crop_size2 = opt.fineSize
        train_crop_size3 = opt.fineSize
        test_size = opt.fineSize

        self.ptrain = os.path.join(opt.workdir, 'train') #'./datasets/Christian-TMR-IF-v0.1/train'
        self.pvalid = os.path.join(opt.workdir, 'valid')
        self.ptest = os.path.join(opt.workdir, 'test') #'./datasets/Christian-TMR-IF-v0.1/test'

        self.input_channels = []
        for ch in opt.input_channels.split(','):
            name, filter = ch.split('=')
            self.input_channels.append((name, {'filter':filter, 'loader':ImageLoader()}, ))

        self.output_channels = []
        for ch in opt.output_channels.split(','):
            name, filter = ch.split('=')
            self.output_channels.append((name, {'filter':filter, 'loader':ImageLoader()}, ))

        # prepare the transforms
        self.iMerge = Merge()
        self.iElastic = ElasticTransform(alpha=1000, sigma=40)
        self.iSplit = Split([0, len(self.input_channels)], [len(self.input_channels), len(self.input_channels)+len(self.output_channels)])

        self.iRCropTrain1 = RandomCropNumpy(size=(train_crop_size1, train_crop_size1))
        self.iRot = RandomRotate()
        self.iCropTrain2 = CenterCropNumpy(size=(train_crop_size2, train_crop_size2))

        self.iCropTest = CenterCropNumpy(size=(test_size, test_size))

        self.dim_ordering = opt.dim_ordering
        self.opt = opt
        self.repeat = 30
        self.input_channel_names = [n for n, _ in self.input_channels]
        self.output_channel_names = [n for n, _ in self.output_channels]

    def __getitem__(self, key):
        if key == 'train':
            source_train = SubfolderDataset(self.ptrain,
                             channels = self.input_channels +  self.output_channels,
                             transform = self.transform_train,
                             repeat=self.repeat)
            return source_train
        elif key == 'valid':
            source_valid = SubfolderDataset(self.pvalid,
                             channels = self.input_channels +  self.output_channels,
                             transform = self.transform_valid,
                             repeat=1)
            return source_valid
        elif key == 'test':
            source_test = SubfolderDataset(self.ptest,
                             channels = self.input_channels,
                             transform = self.transform_test,
                             repeat=1)
            return source_test
        else:
            raise Exception('only train and test are supported.')

    def transform_train(self, images):
        inputs = [np.expand_dims(np.array(images[n]), axis=2) for n in self.input_channel_names]
        outputs = [np.expand_dims(np.array(images[n]), axis=2) for n in self.output_channel_names]
        ios = self.iMerge(inputs + outputs)
        ios = self.iRCropTrain1(ios)
        ios = self.iRot(ios)
        ios = self.iElastic(ios)
        ios = self.iCropTrain2(ios)
        inputs, outputs = self.iSplit(ios)
        return {'A': inputs, 'B': outputs, 'path': images['__path__']}

    def transform_valid(self, images):
        inputs = [np.expand_dims(np.array(images[n]), axis=2) for n in self.input_channel_names]
        outputs = [np.expand_dims(np.array(images[n]), axis=2) for n in self.output_channel_names]
        inputs = self.iCropTest(self.iMerge(inputs))
        outputs = self.iCropTest(self.iMerge(outputs))
        return {'A': inputs, 'B': outputs, 'path': images['__path__']}

    def transform_test(self, images):
        inputs = [np.expand_dims(np.array(images[n]), axis=2) for n in self.input_channel_names]
        inputs = self.iCropTest(self.iMerge(inputs))
        return {'A': inputs, 'path': images['__path__']}


class CompositeRandomDataset():
    def __init__(self, datasets, opt, group='test'):
        datasets = list(datasets.items())
        self.mode = group
        self.opt = opt
        self.dataset_names = [n for n, p in datasets]
        datasets = [(create_data_sources(k, opt) if type(k) is str else k , v) for k, v in datasets ]
        self.datasets = [d for d, p, in datasets]
        self.dataset_ids = list(set([dds.typeID for dds in self.datasets if dds.typeID != -1]))
        self.data_sources = [(d[group], p) for d, p, in datasets]
        self.group = group
        self.lengths = [len(d) for d, p in self.data_sources]
        self.probs = [p for d, p in self.data_sources]
        self.probs_acc = []
        pc = 0
        for p in self.probs:
            pc +=p
            self.probs_acc.append(pc)
        self.probs_max = self.probs_acc[-1]
        self.data_type = None
        self.tags = None
        self.__fpp = None
        self.__repeat = False
        self.__out = None
        self.__additional_source = None
        self.__channel_mask = None
        self.__callback = None
        if self.opt.add_lr_channel:
            self.wfBlur = GaussianBlurring(sigma=['uniform', opt.lr_sigma-1.5, opt.lr_sigma+1.5])
            self.wfNoise = AddGaussianNoise(mean=0, sigma=['uniform', 0.5, 1.5])
        else:
            self.wfBlur = None
            self.wfNoise = None
    def set_callback(self, callback):
        self.__callback = callback

    def set_addtional_source(self, source):
        self.__additional_source = source

    def set_data_type(self, data_type=None):
        if type(data_type) is str:
            typeID = DatasetTypeIDs[data_type]
        else:
            typeID = data_type
        if typeID is not None and self.data_type != typeID:
            print('WARNING: typeID is overrided to: {}({})', data_type, typeID)
        if self.opt.control_classes is not None:
            assert typeID < self.opt.control_classes, 'typeID must be smaller than the control classes number'
        self.data_type = typeID

    def set_tags(self, tags=None):
        self.tags = tags

    def set_fpp(self, fpp=None):
        self.__fpp = fpp

    def set_channel_mask(self, sw=None):
        self.__channel_mask = sw
        assert len(self.__channel_mask) == self.opt.input_nc

    def __getitem__(self, index):
        if self.__repeat and self.__out is not None and self.mode == 'train':
            out = self.__out
            self.__repeat = False
            self.__out = None
            if 'add_fpp_control' in self.opt and self.opt.add_fpp_control:
                out['control'][-1] = 1.0 - out['control'][-1]
        else:
            selected = 0
            c = np.random.random() * self.probs_max
            for i, p in enumerate(self.probs_acc):
                if c<=p:
                    selected = i
                    break
            ds = self.datasets[selected]
            d, _ = self.data_sources[selected]
            if len(self.lengths) == 1:
                new_index = index
            else:
                l = self.lengths[selected]
                new_index = np.random.randint(0, l, 1)[0] #int(1.0 * index/len(self) * l)
            out = d[new_index].copy()
            out['control'] = []
            if self.__additional_source is not None and np.random.random()<0.5:
                nd = self.__additional_source[np.random.randint(0, len(self.__additional_source), 1)[0]]
                if nd['A'].max() <= out['A'].max() or np.random.random()<0.2:
                    # TODO: make it work with multiple channels
                    if 'A' in out and 'A' in nd:
                        out['A'][:, :, :1] += ((nd['A']/nd['A'].max())*(out['A'][:,:,:1].max()*np.random.uniform(0.3, 0.9)))
                    if 'B' in out and 'B' in nd:
                        out['B'][:, :, :1] += nd['B'][:, :, :1]
                    # if 'LR' in out and 'LR' in nd:
                    #     out['LR'] += nd['LR']
            added_empty_channel = False
            if 'add_lr_channel' in self.opt and self.opt.add_lr_channel and out['A'].shape[2] == self.opt.input_nc-1:
                assert self.opt.input_nc > 1
                if self.opt.add_lr_channel == 'empty':
                    out['A'] = np.concatenate([out['A'], np.zeros_like(out['A'])], axis=2)
                    added_empty_channel = True
                elif self.opt.add_lr_channel == 'pseudo':
                    wf = self.wfBlur(scipy.misc.bytescale(out['B']).astype('float32'))[:, :, 0]
                    #wf = scipy.misc.imresize(wf, 0.2)
                    wf = self.wfNoise(wf[:,:,None])
                    #wf = scipy.misc.imresize(wf, out['A'].shape[:2])[:, :, None]
                    out['A'] = np.concatenate([out['A'], wf], axis=2)
                else:
                    raise Exception('lr channel mode error')

            if 'add_data_type_control' in self.opt and self.opt.add_data_type_control:
                typeID = self.data_type if self.data_type is not None else ds.typeID
                if typeID == DatasetTypeIDs['random']:
                    typeID = random.choice(self.dataset_ids)
                if self.opt.control_classes is not None:
                    assert 0 <= typeID < self.opt.control_classes, 'typeID must be smaller than the control classes number'
                # from data.normalization import get_norm
                # print(typeID, get_norm('mean_std')(np.zeros_like(out['A'][0:1, :, :]) + typeID).mean())
                # if ds.dim_ordering == 'channels_first':
                #     out['A'] = np.concatenate([out['A'], np.zeros_like(out['A'][0:1, :, :]) + typeID], axis=0)
                # else:
                #     out['A'] = np.concatenate([out['A'], np.zeros_like(out['A'][:, :, 0:1]) + typeID], axis=2)
                out['control'].append(typeID)
            if 'add_tags_control' in self.opt and self.opt.add_tags_control:
                tags = self.tags if self.tags is not None else ds.tags
                out['tags'] = tags

            # add false positive prevention channel
            if 'add_fpp_control' in self.opt and self.opt.add_fpp_control:
                rw = self.__fpp if self.__fpp is not None else np.random.randint(0, 2)
                # if ds.dim_ordering == 'channels_first':
                #     out['A'] = np.concatenate([out['A'], np.zeros_like(out['A'][0:1, :, :]) + rw], axis=0)
                # else:
                #     out['A'] = np.concatenate([out['A'], np.zeros_like(out['A'][:, :, 0:1]) + rw], axis=2)
                out['control'].append(rw)
                self.__out = out
                self.__repeat = True
            if self.__channel_mask is None and self.opt.use_random_channel_mask and self.opt.input_nc>1:
                _mask = [1]*self.opt.input_nc
                if np.random.random()>0.5 and not added_empty_channel:
                    _sel = np.random.choice(list(range(self.opt.input_nc)))
                    _mask[_sel] = 0
                    print('m', end='')
                out['channel_mask'] = _mask
            else:
                out['channel_mask'] = self.__channel_mask
            if len(out['control']) < self.opt.control_nc:
                out['control'] = out['control'] + [0]*(self.opt.control_nc-len(out['control']))
            assert self.opt.control_nc is None or self.opt.control_nc==0 or len(out['control']) == self.opt.control_nc

        if self.__callback is not None:
            self.__callback(out)

        return out

    def __len__(self):
        return int(np.array(self.lengths).sum())


class CompositeDataset():
    def __init__(self, datasets, opt):
        '''
        dataset = {'TransformedTubulinImages004': 0.3, 'TransformedTubulinImages001': 0.2}
        '''
        if type(datasets) is list:
            datasets = {d: 1.0/len(datasets) for d in datasets}
        else:
            assert type(datasets) is dict
        self.datasets = datasets
        self.opt = opt

    def __getitem__(self, key):
        if key == 'train':
            source_train = CompositeRandomDataset(self.datasets, self.opt, group=key)
            return source_train
        elif key == 'test':
            source_test = CompositeRandomDataset(self.datasets, self.opt, group=key)
            return source_test
        else:
            raise Exception('only train and test are supported.')

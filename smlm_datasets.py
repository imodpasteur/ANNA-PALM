import numpy as np
import scipy
from AnetLib.data.image_utils import RandomRotate, CenterCropNumpy, RandomCropNumpy, PoissonSubsampling, AddGaussianPoissonNoise, GaussianBlurring, AddGaussianNoise, ElasticTransform
from datasets import TUBULIN, NUCLEAR_PORE
from AnetLib.data.image_utils import to_tensor
from AnetLib.data.image_utils import EnhancedCompose, Merge, Split
from AnetLib.data.image_utils import RandomRotate, CenterCropNumpy, RandomCropNumpy
from AnetLib.data.image_utils import NormalizeNumpy, MaxScaleNumpy
from AnetLib.data.folder_dataset import FolderDataset
from palm_utils import generate_image_pairs_from_csv

DatasetTypeIDs = {'random': -1, 'tubulin': 0, 'nuclear_pore': 1, 'actin': 2, 'mitochondria': 3}

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
    elif name == 'TransformedNuclearPore001':
        return TransformedNuclearPore001(opt)
    elif name == 'TransformedNuclearPore001Dense':
        return TransformedNuclearPore001Dense(opt)
    else:
        raise Exception('unsupported dataset')


class TransformedTubulin001():
    def __init__(self, opt):
        self.typeID = DatasetTypeIDs['tubulin']
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

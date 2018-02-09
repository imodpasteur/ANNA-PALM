import numpy as np
import os
import os.path
import scipy
import scipy.ndimage
import collections
from PIL import Image
from AnetLib.data.file_loader import FileLoader
from AnetLib.data.folder_dataset import FolderDataset
from AnetLib.data.image_utils import CenterCropNumpy


LocalizationTable = collections.namedtuple("LocalizationTable", "array, xy_range, z_range, f_range")


def is_numeric(obj):
    attrs = ['__add__', '__sub__', '__mul__', '__pow__'] # python3: '__truediv__', python2: '__div__'
    return all(hasattr(obj, attr) for attr in attrs)


def num_generator(config, index=0, random_state=np.random):
    if config[0] == 'uniform':
        ret = random_state.uniform(config[1], config[2], 1)[0]
    elif config[0] == 'lognormal':
        ret = random_state.lognormal(config[1], config[2], 1)[0]
    elif is_numeric(config[0]):
        ret = config[index]
    else:
        print(is_numeric(config[0]))
        print(config)
        raise Exception('unsupported format')
    return ret


class ThunderstormCSVLoader(FileLoader):
    def __init__(self, xy_range, z_range=None, npz_cache=True, memory_cache=False):
        self.xy_range = xy_range
        self.z_range = z_range
        self.memory_cache = memory_cache
        self.npz_cache = npz_cache

        self.__current_path = None
        self.__current_table = None
        self.__cache = {}

    def load(self, csvFile):
        if csvFile in self.__cache:
            return self.__cache[csvFile]
        try:
            if os.path.exists(csvFile+'.npz'):
                xyfArr = np.load(csvFile+'.npz')['xyfArr']
                return LocalizationTable(array=xyfArr, xy_range=self.xy_range, z_range=self.z_range, f_range=(xyfArr[:, 2].min(), xyfArr[:, 2].max()))
        except:
            pass
        with open(csvFile, "r") as f:
            header = f.readline().split(',')
            fi = [i for i, j in enumerate(header) if 'frame' in j][0]
            xi = [i for i, j in enumerate(header) if 'x[nm]' in j or 'x [nm]' in j][0]
            yi = [i for i, j in enumerate(header) if 'y[nm]' in j or 'y [nm]' in j and 'uncertainty' not in j][0]
        locTable = np.loadtxt(open(csvFile, "rb"), delimiter=",", skiprows=1)
        xyfArr = locTable[:, [xi, yi, fi]]
        xyfArr = xyfArr.astype('int32')
        return LocalizationTable(array=xyfArr, xy_range=self.xy_range, z_range=self.z_range, f_range=(xyfArr[:, 2].min(), xyfArr[:, 2].max()))

    def cache(self, path):
        try:
            table = self.load(path)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            return False
        if self.npz_cache:
            if os.path.exists(path+'.npz'):
                return True
            npzf = path + '.npz'
            np.savez(npzf, xyfArr=table.array)
        if self.memory_cache:
            self.__cache[path] = table
        return True

    def __call__(self, path):
        if self.__current_path == path:
            return self.__current_table
        self.__current_table = self.load(path)
        self.__current_path = path
        return self.__current_table


class LocalizationCrop(object):
    """crop a localization table
    """

    def __init__(self, fit_data=True, top_left=None, crop_size=None):
        self.crop_size = crop_size
        self.top_left = top_left
        self.fit_data = fit_data
        # self.size_base2 = size_base2

    def __call__(self, table):
        xyfArr = table.array
        if xyfArr.shape[0] == 0:
            return table
        x = xyfArr[:, 0]
        y = xyfArr[:, 1]

        if self.fit_data:
            xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
            if self.top_left:
                ox, oy = self.top_left
            else:
                ox, oy = xmin, ymin
            xsize, ysize = xmax-ox, ymax-oy
            # if self.size_base2:
            #     xsize = 2**int(np.ceil(np.log2(xsize)))
            #     ysize = 2**int(np.ceil(np.log2(ysize)))
        else:
            assert self.crop_size, 'crop_size must be set'
            xsize = self.crop_size[0]
            ysize = self.crop_size[1]
            if self.top_left:
                ox, oy = self.top_left
            else:
                ox, oy = (0, 0)
        cx = np.logical_and(xyfArr[:, 0] >= ox, xyfArr[:, 0] <= ox+xsize)
        cy = np.logical_and(xyfArr[:, 1] >= oy, xyfArr[:, 1] <= oy+ysize)
        cxy = np.logical_and(cx, cy)
        xy_range = [ox, ox+xsize, oy, oy+ysize]
        return LocalizationTable(array=xyfArr[cxy, :], xy_range=xy_range, z_range=table.z_range, f_range=(xyfArr[:, 2].min(), xyfArr[:, 2].max()))


class LocalizationRandomCrop(object):
    """crop a localization table
    """

    def __init__(self, crop_size, no_blank=True, max_try=10):
        self.crop_size = crop_size
        self.no_blank = no_blank
        self.max_try = max_try

    def __call__(self, table):
        xyfArr = table.array
        if xyfArr.shape[0] == 0:
            return table
        retSum = 0
        retry = 0
        no_blank = self.no_blank
        while retSum == 0:
            # x = xyfArr[:, 0]
            # y = xyfArr[:, 1]
            xmin, xmax, ymin, ymax = table.xy_range  # x.min(),x.max(),y.min(),y.max()
            if xmax-self.crop_size[0] <= xmin or ymax-self.crop_size[1] <= ymin:
                ox, oy = xmin, ymin
                # disable no blank
                no_blank = False
            else:
                ox, oy = np.random.uniform(xmin, xmax-self.crop_size[0]), np.random.uniform(ymin, ymax-self.crop_size[1])
            cx = np.logical_and(xyfArr[:, 0] >= ox, xyfArr[:, 0] < ox+self.crop_size[0])
            cy = np.logical_and(xyfArr[:, 1] >= oy, xyfArr[:, 1] < oy+self.crop_size[1])
            cxy = np.logical_and(cx, cy)
            xy_range = [ox, ox+self.crop_size[0], oy, oy+self.crop_size[1]]
            retArr = xyfArr[cxy, :]
            if not no_blank:
                break
            retSum = retArr.sum()
            retry += 1
            if self.max_try and retry > self.max_try:
                break

        return LocalizationTable(array=retArr, xy_range=xy_range, z_range=table.z_range, f_range=(retArr[:, 2].min(), retArr[:, 2].max()))


class LocalizationFrameSampler(object):
    """Sample a localization table
    """

    def __init__(self, frame_num, zero_offset=False, frame_limit=None):
        self.frame_num = frame_num
        self.zero_offset = zero_offset
        self.frame_limit = frame_limit

    def __call__(self, table, index=0):
        frame_num = self.frame_num
        if isinstance(frame_num, collections.Sequence):
            frame_num = int(num_generator(frame_num, index))
        assert frame_num >= 0
        xyfArr = table.array
        if xyfArr.shape[0] == 0:
            return table
        fmax = xyfArr[:, 2].max()
        fmin = xyfArr[:, 2].min()
        if self.frame_limit is not None:
            self.frame_limit = [fmax*f if 0 < f <= 1 else f for f in self.frame_limit]
            assert self.frame_limit[0]<self.frame_limit[1]
            fmin = max(self.frame_limit[0], fmin)
            fmax = min(fmax, self.frame_limit[1])

        frame_num = int((fmax*frame_num if 0 < frame_num <= 1 else frame_num)+0.5)

        if self.zero_offset:
            if fmax-frame_num <= fmin:
                frame_num = fmax-fmin
            ofout = fmin
        else:
            if fmax-frame_num <= fmin:
                frame_num = fmax-fmin
                ofout = fmin
            else:
                ofout = np.random.randint(fmin, fmax-frame_num)

        cfout = np.logical_and(xyfArr[:,2] >= ofout, xyfArr[:, 2] < ofout+frame_num)
        return LocalizationTable(array=xyfArr[cfout, :], xy_range=table.xy_range, z_range=table.z_range, f_range=(ofout, ofout+frame_num))


class HistogramRendering(object):
    """Render a localization table to a histogram image
    """

    def __init__(self, pixel_size, value_range=None, sigma=None, target_size=None):
        self.value_range = value_range
        self.pixel_size = pixel_size
        self.sigma = sigma
        self.target_size = target_size

    def __call__(self, table):
        xyArr = table.array
        x = xyArr[:, 0]
        y = xyArr[:, 1]
        if table.xy_range:
            xmin, xmax, ymin, ymax = table.xy_range
        else:
            xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
        xedges = np.arange(xmin, xmax+0.5, self.pixel_size)
        yedges = np.arange(ymin, ymax+0.5, self.pixel_size)
        H, xedgesO, yedgesO = np.histogram2d(y, x, bins=(yedges, xedges))
        if self.target_size is not None:
            if H.shape[0] < self.target_size[0]:
                H = np.pad(H, ((0, self.target_size[0]-H.shape[0]), (0, 0)), mode='constant', constant_values=0)
            if H.shape[1] < self.target_size[1]:
                H = np.pad(H, ((0, 0), (0, self.target_size[1]-H.shape[1])), mode='constant', constant_values=0)
            H = H[: self.target_size[0], : self.target_size[1]]
        if self.value_range:
            H = H.clip(self.value_range[0], self.value_range[1])
        if self.sigma:
            H = scipy.ndimage.filters.gaussian_filter(H, sigma=(self.sigma, self.sigma))
        return H[:, :, None]


class SubFolderImagesLoader(FileLoader):
    def __init__(self, drift_correction=True, extension='png'):
        self.__cache = {}
        self.ext = extension
        self.NoiseImages = None
        self.drift_correction = drift_correction

    def load(self, path):
        if path not in self.__cache:
            self.cache(path)
        return self.__cache[path]

    def save_cache(self, path):
        path = os.path.join(path, 'noise')
        if os.path.exists(path):
            Ns = [os.path.join(path, p) for p in os.listdir(path) if p.endswith(self.ext)]
            ImgNs = []
            for p in Ns:
                try:
                    ImgNs.append(np.array(Image.open(p)))
                except Exception as e:
                    print('error when reading noise file ', p)
            if len(ImgNs)>0:
                self.NoiseImages = ImgNs
            else:
                self.NoiseImages = None
        else:
            self.NoiseImages = None
        for path in self.__cache.keys():
            self.__cache[path].update({'N': self.NoiseImages})


    def cache(self, path):
        As = [os.path.join(path, p) for p in os.listdir(path) if p.startswith('A') and p.endswith(self.ext)]
        Bs = [os.path.join(path, p) for p in os.listdir(path) if p.startswith('B') and p.endswith(self.ext)]
        LRs = [os.path.join(path, p) for p in os.listdir(path) if p.startswith('LR') and p.endswith(self.ext)]
        if os.path.exists(os.path.join('mask_A' + self.ext)):
            m = np.array(Image.open(os.path.join('mask_A' + self.ext)))
            m = np.expand_dims(m, axis=2) if m.ndim == 2 else m
            maskA = m < m.max()/2
        else:
            maskA = None
        if os.path.exists(os.path.join('mask_B' + self.ext)):
            m = np.array(Image.open(os.path.join('mask_B' + self.ext)))
            m = np.expand_dims(m, axis=2) if m.ndim == 2 else m
            maskB = m < m.max()/2
        else:
            maskB = None
        ImgAs, PathAs, ImgBs, PathBs, ImgLRs, PathLRs= [], [], [], [], [], []
        for p in As:
            try:
                img = np.array(Image.open(p))
                img = np.expand_dims(img, axis=2) if img.ndim == 2 else img
                if maskA:
                    img[maskA] = img.min()
                ImgAs.append(img)
                PathAs.append(p)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print('error when reading file ', p)
        assert len(ImgAs) > 0, 'no file found for "A"'
        for p in Bs:
            try:
                img = np.array(Image.open(p))
                img = np.expand_dims(img, axis=2) if img.ndim == 2 else img
                if maskB:
                    img[maskB] = img.min()
                ImgBs.append(img)
                PathBs.append(p)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print('error when reading file ', p)

        for p in LRs:
            try:
                img = np.array(Image.open(p))
                assert img.ndim == 2
                if self.drift_correction:
                    import imreg_dft as ird
                    from skimage import exposure
                    b = ImgBs[0][:, :, 0]
                    b = exposure.equalize_hist(b)
                    b = scipy.ndimage.filters.gaussian_filter(b, sigma=(6, 6))
                    b = scipy.misc.imresize(b, img.shape[:2])
                    ts = ird.translation(b, img)
                    tvec = ts["tvec"].round(4)
                    # the Transformed IMaGe.
                    img = ird.transform_img(img, tvec=tvec)
                img = scipy.misc.imresize(img, ImgBs[0].shape[:2])
                img = np.expand_dims(img, axis=2)
                ImgLRs.append(img)
                PathLRs.append(p)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print('error when reading file ', p)
                import traceback, sys
                traceback.print_exc(file=sys.stdout)

        self.__cache[path] = {'A': ImgAs, 'B': ImgBs, 'LR':ImgLRs, 'path': path, 'pathA': PathAs, 'pathB': PathBs, 'pathLR': PathLRs}
        return True

    def __call__(self, path):
        if path not in self.__cache:
            self.cache(path)
        return self.__cache[path].copy()

# A_frame_limit=[0, 6000], B_frame_limit=[2000, 1.0]
def generate_image_pairs_from_csv(csv_folder, output_dir, image_per_file=10, A_frame=150, B_frame=0.85, file_filter='*.csv',
                                  top_left=(0, 0), input_size_nm=512*106, pixel_size=20, A_frame_limit=[0, 1.0], B_frame_limit=[0, 1.0],
                                  output_clip = (0, 255), input_clip = (0, 20), target_size=(2560, 2560), center_crop=None, zero_offset=False):
    lCropTrain = LocalizationCrop(fit_data=True, top_left=top_left)
    fSamplerInTest = LocalizationFrameSampler(frame_num=A_frame, frame_limit=A_frame_limit, zero_offset=zero_offset)
    fSamplerOutTest = LocalizationFrameSampler(frame_num=B_frame, frame_limit=B_frame_limit)
    hRender = HistogramRendering(pixel_size=pixel_size, value_range= (0, 255), target_size=target_size)
    if center_crop:
        cropTest = CenterCropNumpy(size=center_crop)
    def transform_train(imgDict):
        table = imgDict['table']
        repeat = imgDict['table.repeat']
        table = lCropTrain(table)
        tableout = fSamplerOutTest(table, index=repeat)
        tablein = fSamplerInTest(table, index=repeat)
        histout = hRender(tableout)
        histin = hRender(tablein)
        if center_crop:
            histin = cropTest(histin)
            histout = cropTest(histout)
        histin = np.clip(histin, 0, 255)
        histout = np.clip(histout, 0, 255)
        return histin, histout, imgDict['table.path'], tablein.f_range, tableout.f_range

    csvLoader = ThunderstormCSVLoader([0, input_size_nm, 0, input_size_nm])
    source_train = FolderDataset(csv_folder,
                      channels = {'table': {'filter':file_filter, 'loader': csvLoader} },
                     transform = transform_train,
                     recursive=False,
                     repeat=image_per_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('generating images...')
    for i in range(len(source_train)):
        (aa, bb, p, inf_range, outf_range) = source_train[i]
        print(i, inf_range, outf_range)
        name = os.path.split(p)[1]
        fpath = os.path.join(output_dir, name)
        if not os.path.exists(fpath):
            os.makedirs(fpath)

        ha = aa[:,:,0].astype('uint8')
        imA = Image.fromarray(ha)
        imA.save(os.path.join(fpath, 'A_{}_{}.png'.format(i%image_per_file, str(inf_range))))

        hb = bb[:,:,0].astype('uint8')
        imB = Image.fromarray(hb)
        if B_frame == 1.0:
            imB.save(os.path.join(fpath, 'B_{}_{}.png'.format('all', str(outf_range))))
        else:
            imB.save(os.path.join(fpath, 'B_{}_{}.png'.format(i%image_per_file, str(outf_range))))
    print('done')

def generate_images_from_csv(csv_folder, output_dir, frame=1.0, image_per_file=1, file_filter='*.csv', zero_offset=False,
                             top_left=(0, 0), input_size_nm=512*106, pixel_size=20, frame_limit=[0, 1.0],
                             output_clip = (0, 255), input_clip = (0, 20), target_size=(2560, 2560), center_crop=None):
    lCropTrain = LocalizationCrop(fit_data=True, top_left=top_left)
    fSamplerOutTest = LocalizationFrameSampler(frame_num=frame, frame_limit=frame_limit, zero_offset=zero_offset)
    hRender = HistogramRendering(pixel_size=pixel_size, value_range= (0, 255), target_size=target_size)
    if center_crop:
        cropTest = CenterCropNumpy(size=center_crop)
    def transform_train(imgDict):
        table = imgDict['table']
        table = lCropTrain(table)
        repeat = imgDict['table.repeat']
        path = imgDict['table.path']
        tableout = fSamplerOutTest(table, index=repeat)
        path = path + '_' + str(repeat)+ '_' + str(tableout.f_range)
        histout = hRender(tableout)
        if center_crop:
            histout = cropTest(histout)
        histout = np.clip(histout, 0, 255)
        return histout, path

    csvLoader = ThunderstormCSVLoader([0, input_size_nm, 0, input_size_nm])
    source_train = FolderDataset(csv_folder,
                      channels = {'table': {'filter':file_filter, 'loader': csvLoader} },
                     transform = transform_train,
                     repeat=image_per_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('generating images...')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(len(source_train)):
        (bb, p) = source_train[i]
        print(i, p)
        name = os.path.split(p)[1]
        fpath = os.path.join(output_dir, name + '_{}.png'.format(frame))
        hb = bb[:,:,0].astype('uint8')
        imB = Image.fromarray(hb)
        imB.save(fpath)
    print('done')


def generate_noise_from_mask(image_folder, mask_file='mask_B.png', window=40, overlap=10):
    subfolders = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if not f.startswith('.') and os.path.isdir(os.path.join(image_folder, f))]
    print('generating noise image...')
    for folder in subfolders:
        As = [os.path.join(folder, p) for p in os.listdir(folder) if p.startswith('A')]
        Bs = [os.path.join(folder, p) for p in os.listdir(folder) if p.startswith('B')]
        if os.path.exists(os.path.join(folder, mask_file)):
            mask = np.array(Image.open(os.path.join(folder, mask_file)))
            rwm = rolling_window(mask, (window, window), asteps=(window-overlap, window-overlap))
            coords = rwm.sum(axis=(2,3)) ==0 # < (window*window*mask.max()/2)
            noise_stack = np.zeros((0, window, window))
            for bp in As+Bs:
                bdir, bname = os.path.split(bp)
                imgB = np.array(Image.open(bp))
                rw = rolling_window(imgB, (window, window), asteps=(window-overlap, window-overlap))
                coords2 = rw.sum(axis=(2,3)) > 0
                m = rw[np.logical_and(coords, coords2)]
                noise_stack = np.concatenate([noise_stack, m])
                print('.', end='')
            ss = np.argsort(m.sum(axis=(1,2)))
            m = m[ss]
            mr = np.rot90(m, axes=(1,2))
            h, w = coords.shape[0], coords.shape[1]
            num = int(m.shape[0]//2*2)
            print(num)
            cc = 5
            if not os.path.exists(os.path.join(image_folder,'noise')):
                os.makedirs(os.path.join(image_folder,'noise'))
            for i in range(cc):
                start = i*num//cc
                selected = np.random.randint(start,start+(num//cc), int(w*h)//2)
                selectedr = np.random.randint(start,start+(num//cc), int(w*h)-int(w*h)//2)
                noise_img = np.concatenate([m[selected, :, :], mr[selectedr, :, :]])
                noise_img = noise_img.reshape(h, w, window, window).transpose([0, 2, 1, 3]).reshape(h*window, w*window)
                noise_img = Image.fromarray(noise_img.astype(imgB.dtype))
                noise_img = noise_img.crop((0, 0, mask.shape[0], mask.shape[1]))
                dd,ff = os.path.split(folder)
                noise_img.save(os.path.join(image_folder,'noise', 'A_noise_'+ff+'_'+str(i)+'.png'))
            Image.fromarray(np.array(noise_img)*0).save(os.path.join(image_folder,'noise', 'B_noise_empty.png'))
    print('done')

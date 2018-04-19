import zipfile
import json
import struct
import io
import numpy as np
from PIL import Image

dtype2struct = {'uint8': 'B', 'uint32': 'I', 'float64': 'd', 'float32': 'f'}
def import_smlm(file_path):
    zf = zipfile.ZipFile(file_path, 'r')
    file_names = zf.namelist()
    if "manifest.json" in file_names:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest['format_version'] == '0.2'
        for file_info in manifest['files']:
            format_key = file_info['format']
            file_format = manifest['formats'][format_key]
            if file_info['type'] == "table":
                print('loading table...')
                if file_format['mode'] == 'binary':
                    try:
                        table_file = zf.read(file_info['name'])
                        print(file_info['name'])
                    except KeyError:
                        print('ERROR: Did not find %s in zip file' % file_info['name'])
                        continue
                    else:
                        print('loading table file: {} bytes'.format(len(table_file)))
                        print('file format: ', file_format)
                        headers = file_format['headers']
                        dtype = file_format['dtype']
                        shape = file_format['shape']
                        hLen = len(headers)
                        assert len(headers) == len(headers) == len(shape)
                        st = ''
                        for i, h in enumerate(file_format['headers']):
                            st += (str(shape[i])+dtype2struct[dtype[i]])
                        rowLen = struct.calcsize(st)
                        s = struct.Struct(st)
                        tableDict = {h:[] for h in headers}
                        for i in range(0, len(table_file), rowLen):
                            unpacked_data = s.unpack_from(table_file, i)
                            for j, h in enumerate(headers):
                                tableDict[h].append(unpacked_data[j])
                        tableDict = {h:np.array(tableDict[h]) for i,h in enumerate(headers)}
                        data = {}
                        data['min'] = [tableDict[h].min() for h in headers]
                        data['max'] = [tableDict[h].max() for h in headers]
                        data['avg'] = [tableDict[h].mean() for h in headers]
                        data['tableDict'] = tableDict
                        file_info['data'] = data
                        print('table file loaded: ', file_info['name'])
                else:
                    raise Exception('format mode {} not supported yet'.format(file_format['mode']))
            elif file_info['type'] == "image":
                if file_format['mode'] == 'binary':
                    try:
                        image_file = zf.read(file_info['name'])
                        print(file_info['name'])
                    except KeyError:
                        print('ERROR: Did not find %s in zip file' % file_info['name'])
                        continue
                    else:
                        image = Image.open(io.BytesIO(image_file))
                        data = {}
                        data['image'] = image
                        file_info['data'] = data
                        print('image file loaded: ', file_info['name'])

            else:
                print('ignore file with type: ', file_info['type'])
    else:
        raise Exception('invalid file: no manifest.json found in the smlm file')
    return manifest, manifest['files']

def plotHist(tableDict, value_range=None, xy_range=None, pixel_size=20, sigma=None, target_size=None):
    x = tableDict['x'][:]
    y = tableDict['y'][:]
    if xy_range:
        xmin, xmax = xy_range[0]
        ymin, ymax = xy_range[1]
    else:
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    xedges = np.arange(xmin, xmax, pixel_size)
    yedges = np.arange(ymin, ymax, pixel_size)
    H, xedgesO, yedgesO = np.histogram2d(y, x, bins=(yedges, xedges))
    if target_size is not None:
        if H.shape[0] < target_size[0] or H.shape[1] < target_size[1]:
            H = np.pad(H, ((0, target_size[0] - H.shape[0]), (0, target_size[
                       1] - H.shape[1])), mode='constant', constant_values=0)

    if value_range:
        H = H.clip(value_range[0], value_range[1])
    if sigma:
        H = scipy.ndimage.filters.gaussian_filter(H, sigma=(sigma, sigma))
    return H

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    manifest, files = import_smlm('../../xxxx.smlm')
    h = plotHist(files[0]['data']['tableDict'], value_range=(0,10))
    plt.figure(figsize=(20,20))
    plt.imshow(h)
    plt.figure(figsize=(20,20))
    plt.imshow(np.array(files[1]['data']['image']))

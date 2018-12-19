import os
import random
import shutil
from localization_utils import generate_image_pairs_from_csv, SubFolderImagesLoader

train_datadir = '../train'
test_datadir = '../valid'
workdir = '../workdirs'
seed = 123

random.seed(seed)

train_files = sorted([os.path.join(train_datadir, f) for f in os.listdir(train_datadir) if f.endswith('.csv')])
indexes = list(range(len(train_files)))
random.shuffle(indexes)
test_files = sorted([os.path.join(test_datadir, f) for f in os.listdir(test_datadir) if f.endswith('.csv')])
print('shuffled indexes:', ','.join(map(str, indexes)))


if not os.path.exists(workdir):
    os.makedirs(workdir)

root_test_img_workdir = os.path.join(workdir, 'test')
# if not os.path.exists(root_test_img_workdir):
#     os.makedirs(root_test_img_workdir)
# else:
#     shutil.rmtree(root_test_img_workdir)
#     os.makedirs(root_test_img_workdir)
#
# print('generating test data.')
# generate_image_pairs_from_csv(test_datadir,
#                         root_test_img_workdir,
#                         A_frame=['uniform', 300, 400], B_frame=1.0,
#                         A_frame_limit=(0, 1.0),
#                         B_frame_limit=(0, 1.0),
#                         image_per_file=5,
#                         target_size=None,
#                         zero_offset=True)

for num in [1, 2, 4, 8, 16, 32, 64]:
    print('processing ', num)
    train_workdir = os.path.join(workdir, 'workdir-'+str(num), 'train_csv')
    # test_workdir = os.path.join(workdir, 'workdir-'+str(num), 'test_csv')
    if not os.path.exists(train_workdir):
        os.makedirs(train_workdir)
    else:
        shutil.rmtree(train_workdir)
        os.makedirs(train_workdir)

    # if not os.path.exists(test_workdir):
    #     os.makedirs(test_workdir)
    # else:
    #     shutil.rmtree(test_workdir)
    #     os.makedirs(test_workdir)

    train_img_workdir = os.path.join(workdir, 'workdir-'+str(num), 'train')
    test_img_workdir = os.path.join(workdir, 'workdir-'+str(num), 'test')
    if not os.path.exists(train_img_workdir):
        os.makedirs(train_img_workdir)
    else:
        shutil.rmtree(train_img_workdir)
        os.makedirs(train_img_workdir)


    # for f in test_files:
    #     _, name = os.path.split(f)
    #     os.symlink(os.path.abspath(f), os.path.join(test_workdir, name))

    if not os.path.exists(test_img_workdir):
        shutil.copytree(os.path.abspath(root_test_img_workdir), test_img_workdir)
    else:
        os.unlink(test_img_workdir)
        shutil.copytree(os.path.abspath(root_test_img_workdir), test_img_workdir)

    for i in indexes[:num]:
        f = train_files[i]
        _, name = os.path.split(f)
        os.symlink(os.path.abspath(f), os.path.join(train_workdir, name))

    generate_image_pairs_from_csv(train_workdir,
                            train_img_workdir,
                            A_frame=['uniform', 300, 400], B_frame=1.0,
                            A_frame_limit=(0, 1.0),
                            B_frame_limit=(0, 1.0),
                            image_per_file=5,
                            target_size=None)

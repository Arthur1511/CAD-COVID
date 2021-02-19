import torch.utils.data as data
import os
import numpy as np
import torch
import pandas as pd

from skimage import transform
from skimage import io

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def default_loader(path):

    return io.imread(path)


def get_img_list(dir, dataset_number, csv_file, mode):
    img_list = []
    label_list = []

    img_dir = os.path.join(dir, 'dataset' + dataset_number + 'images/')
    csv_labels = str(dir + 'dataset' + dataset_number + '/' + csv_file)
    labels = pd.read_csv(csv_labels, usecols=['Image_ID', 'Labels'])

    tam_dataset = labels.shape[0]

    if mode == 'train':
        tam_list = int(0.8*tam_dataset)
        labels = labels.iloc[0:tam_list]
    else:
        tam_list = int(0.2*tam_dataset)
        labels = labels.iloc[-tam_list:]

    for i in range(tam_list):
        lab = labels.iloc[i]['Labels']
        img = labels.iloc[i]['Image_ID']
        try:
            lab = lab.split('|')
        except:
            lab = ['No Finding']

        if not all(x in lab for x in ['Cardiomegaly', 'Atelectasis']):
            label_list.append(lab)
            img_list.append(img)

    for j in range(len(label_list)):
        if 'Cardiomegaly' in label_list[j]:
            label_list[j] = [0]
        elif 'Atelectasis' in label_list[j]:
            label_list[j] = [1]
        else:
            label_list[j] = [2]

    return img_list, label_list


'''
list_im, list_label = get_img_list('/home/CADCOVID/Datasets_CoDAGANs/', 'dataset3/', 'dataset3.csv', 'test')
print(list_im)
print(list_label)
'''


class ImageFolder(data.Dataset):

    def __init__(self, root, sample, dataset_number='0', mode='train',  loader=default_loader,  trim_bool=0, return_path=False, random_transform=False, channels=1, normalization='minmax'):
        file_csv = str('dataset' + dataset_number + '.csv')
        # file_txt = str(mode + '.txt')
        imgs, labels = get_img_list(root, dataset_number, file_csv, mode)

        self.root = root
        self.fold = 'dataset' + dataset_number
        self.imgs = imgs
        self.labels = labels
        self.loader = loader
        self.sample = sample
        self.trim_bool = trim_bool
        self.return_path = return_path
        self.random_transform = random_transform
        self.channels = channels
        self.normalization = normalization

        np.random.seed(12345)

        perm = np.random.permutation(len(imgs))
        self.has_label = np.zeros((len(imgs)), np.int)
        self.has_label[perm[0:int(self.sample * len(imgs))]] = 1

    def trim(self, img):

        tolerance = 0.05 * float(img.max())

        # Mask of non-black pixels (assuming image has a single channel).
        bin = img > tolerance

        # Coordinates of non-black pixels.
        coords = np.argwhere(bin)

        # Bounding box of non-black pixels.
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

        # Get the contents of the bounding box.
        img_crop = img[x0:x1, y0:y1]

        return img_crop

    # Data augmentation.
    def transform(self, img, negate=True, max_angle=8, low=0.1, high=0.9, shear=0.0, fliplr=False, flipud=False):

        # Random color inversion.
        if negate:
            if np.random.uniform() > 0.5:
                if self.channels == 1:
                    img = (img.max() - img)
                else:
                    for i in range(img.shape[2]):
                        img[:, :, i] = (img[:, :, i].max() - img[:, :, i])

        # Random Flipping.
        if fliplr:
            if np.random.uniform() > 0.5:
                img = np.fliplr(img)

        if flipud:
            if np.random.uniform() > 0.5:
                img = np.flipud(img)

        # Random Rotation.
        if max_angle != 0.0:
            angle = np.random.uniform() * max_angle
            if np.random.uniform() > 0.5:
                angle = angle * -1.0

            img = transform.rotate(img, angle, resize=False)

        # Random Shear.
        if shear != 0.0:
            rand_shear = np.random.uniform(low=0, high=shear)
            affine = transform.AffineTransform(shear=rand_shear)

            img = transform.warp(img, inverse_map=affine)

        # Crop.
        if low != 0.0 or high != 1.0:
            beg_crop = np.random.uniform(low=0, high=low, size=2)
            end_crop = np.random.uniform(low=high, high=1.0, size=2)

            s0 = img.shape[0]
            s1 = img.shape[1]

            img = img[int(beg_crop[0] * s0):int(end_crop[0] * s0),
                      int(beg_crop[1] * s1):int(end_crop[1] * s1)]

        return img

    def __getitem__(self, index):

        item = self.imgs[index]
        lbl = self.labels[index]
        lbl = lbl[0]

        file_name = str(self.root + self.fold + '/images/' + item)

        img = self.loader(file_name)

        if self.channels == 1:
            if len(img.shape) > 2:
                img = img[:, :, 0]

        #img = transform.resize(img, msk.shape, preserve_range=True)

        resize_to = (256, 256)

        use_label = False

        if self.trim_bool != 0:
            img = self.trim(img)

        if self.random_transform == 3:
            img = self.transform(img, negate=False, max_angle=90,
                                 low=0.2, high=0.8, shear=0.05, fliplr=True, flipud=True)
        elif self.random_transform == 2:
            img = self.transform(img)
        elif self.random_transform == 1:
            img = self.transform(
                img, negate=False, max_angle=2, low=0.05, high=0.95)

        if self.sample != -1:
            if self.has_label[index] != 0:
                use_label = True
            else:
                use_label = False
        else:
            use_label = True

        if not use_label:
            lbl[:] = -1

        img = transform.resize(
            img, resize_to, preserve_range=True).astype(np.float32)
        if self.channels == 1:

            #img = (img - img.mean()) / (img.std() + 1e-10)
            if self.normalization == 'minmax':
                mn = img.min()
                mx = img.max()
                img = (img - mn) / (mx - mn + 1e-10)
                img = np.expand_dims(img, 0)
            elif self.normalization == 'statistical':
                img = (img - img.mean()) / (img.std() + 1e-10)
                img = np.expand_dims(img, 0)

        else:

            tmp = np.zeros(
                (img.shape[2], img.shape[0], img.shape[1]), dtype=np.float32)

            for i in range(img.shape[2]):

                tmp[i, :, :] = img[:, :, i]

                if self.normalization == 'minmax':
                    mn = tmp[i, :, :].min()
                    mx = tmp[i, :, :].max()
                    tmp[i, :, :] = (tmp[i, :, :] - mn) / (mx - mn + 1e-10)

                if self.normalization == 'statistical':
                    tmp[i, :, :] = (tmp[i, :, :] - tmp[i, :, :].mean()
                                    ) / (tmp[i, :, :].std() + 1e-10)

            img = tmp

        img = torch.from_numpy(img)

        if self.return_path:
            return img, lbl, use_label, file_name
        else:
            return img, lbl, use_label

    def __len__(self):

        return len(self.imgs)


'''
root = '/home/CADCOVID/Datasets_CoDAGANs/'
dataset = ImageFolder(root, fold='dataset1', mode='train', loader=default_loader, trim_bool=0, return_path=False, random_transform=False, channels=1, normalization='minmax')

print(dataset.__getitem__(0))
'''

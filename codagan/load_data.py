import os

import numpy as np
import pandas as pd
import torch
from skimage import transform
from skimage import io
from sklearn.utils import shuffle
from torch.utils.data import Dataset

path = "/home/arthur/CADCOVID/chest-x-ray-8/"

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def default_loader(path):

    return io.imread(path)


def make_dataset(dir, data, classes, split_idx, fold):

    items = []

    img_dir = os.path.join(dir, "images", "images")
    print("img dir: ", img_dir)

    assert os.path.isdir(img_dir), '%s is not a valid directory' % dir
    # if sample > 0.0:

    if fold == 'train':
        for file, img_label in zip(data["Image Index"].loc[:split_idx], data["Finding Labels"].loc[:split_idx]):
            img_path = os.path.join(img_dir, file)

            label = [0] * 15
            labs = img_label.split("|")

            for l in labs:
                label[classes[l]] = 1

            items.append({'img': img_path, 'lbl': label, 'file': file})

    else:
        for file, img_label in zip(data["Image Index"].loc[split_idx:], data["Finding Labels"].loc[split_idx:]):
            img_path = os.path.join(img_dir, file)

            label = [0] * 15
            labs = img_label.split("|")

            for l in labs:
                label[classes[l]] = 1

            items.append({'img': img_path, 'lbl': label, 'file': file})

    return items


class ChestXray14_Dataset(Dataset):

    def __init__(self, dir, sample, fold='train', loader=default_loader, trim_bool=0, return_path=False, random_transform=False, channels=1, normalization='minmax'):

        # Create dict of classes
        self.classes = {
            "Atelectasis": 0,
            "Cardiomegaly": 1,
            "Effusion": 2,
            "Infiltration": 3,
            "Mass": 4,
            "Nodule": 5,
            "Pneumonia": 6,
            "Pneumothorax": 7,
            "Consolidation": 8,
            "Edema": 9,
            "Emphysema": 10,
            "Fibrosis": 11,
            "Pleural_Thickening": 12,
            "Hernia": 13,
            "No Finding": 14
        }

        self.data = pd.read_csv(
            dir + "Data_Entry_2017_v2020.csv")[["Image Index", "Finding Labels"]]
        self.split_idx = int(self.data.shape[0] * 0.8)
        imgs = make_dataset(dir, self.data, self.classes, self.split_idx, fold)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = dir
        self.imgs = imgs
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

    ############################################################################################################################
    # Trim function adapted from: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy #
    ############################################################################################################################
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

        img_path = item['img']
        label = np.array(item['lbl'])
        file_name = item['file']

        img = self.loader(img_path)
        # if self.sample > 0.0:

        if self.channels == 1:
            if len(img.shape) > 2:
                img = img[:, :, 0]

        resize_to = (256, 256)

        # img = transform.resize(img, resize_to, preserve_range=True)

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
            label[:] = 0

        img = transform.resize(
            img, resize_to, preserve_range=True).astype(np.float32)
        if self.channels == 1:

            # img = (img - img.mean()) / (img.std() + 1e-10)
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

        label = torch.from_numpy(label)

        if self.return_path:
            return img, label, use_label, file_name
        else:
            return img, label, use_label

    def __len__(self):

        return len(self.imgs)


dataset = ChestXray14_Dataset(
    path, sample=1, fold='train', trim_bool=True, return_path=True, random_transform=False, channels=1)

print(dataset[0])

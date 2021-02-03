"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, norm
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
# from tester import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # Will be 3.x series.
    pass
import os
import sys
import math
import shutil
import numpy as np

from skimage import io

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/CXR_lungs_MUNIT_1.0.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--load', type=int, default=400)
parser.add_argument('--snapshot_dir', type=str, default='.')
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting.
config = get_config(opts.config)

# Setup model and data loader.
if config['trainer'] == 'MUNIT':
    trainer = MUNIT_Trainer(config, resume_epoch=opts.load, snapshot_dir=opts.snapshot_dir)
elif config['trainer'] == 'UNIT':
    trainer = UNIT_Trainer(config, resume_epoch=opts.load, snapshot_dir=opts.snapshot_dir)
else:
    sys.exit("Only support MUNIT|UNIT.")
    os.exit()

trainer.cuda()

dataset_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
samples = list()
dataset_probs = list()
augmentation = list()
for i in range(config['n_datasets']):
    samples.append(config['sample_' + dataset_letters[i]])
    dataset_probs.append(config['prob_' + dataset_letters[i]])
    augmentation.append(config['transform_' + dataset_letters[i]])

_, test_loader_list = get_all_data_loaders(config, config['n_datasets'], samples, augmentation, config['trim'])

# Setup logger and output folders.
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # Copy config file to output folder.

# Creating isomorphic directory.
if not os.path.exists(os.path.join(image_directory, 'translate')):
    os.mkdir(os.path.join(image_directory, 'translate'))

# Start test.
for i in range(1, config['n_datasets']):

    print('    Testing ' + dataset_letters[i] + '...')

    jacc_list = list()
    for it, (data_src, data_trg) in enumerate(zip(test_loader_list[0], test_loader_list[i])):

        images_src = data_src[0]
        labels_src = data_src[1]
        use_src = data_src[2]
        path_src = data_src[3]
        
        images_trg = data_trg[0]
        labels_trg = data_trg[1]
        use_trg = data_trg[2]
        path_trg = data_trg[3]

        images_src = Variable(images_src.cuda())
#         images_src = Variable(images_src)

        images_trg = Variable(images_trg.cuda())
#         images_trg = Variable(images_trg)

        labels_src = labels_src.to(dtype=torch.long)
        labels_src[labels_src > 0] = 1
        labels_src = Variable(labels_src.cuda(), requires_grad=False)
#         labels_src = Variable(labels_src, requires_grad=False)

        labels_trg = labels_trg.to(dtype=torch.long)
        labels_trg[labels_trg > 0] = 1
        labels_trg = Variable(labels_trg.cuda(), requires_grad=False)
#         labels_trg = Variable(labels_trg, requires_grad=False)

        synths_src, synths_trg = trainer.translate(images_src, images_trg, 0, i, config)

        images_src_path = os.path.join(image_directory, 'translate', path_src[0])
        images_trg_path = os.path.join(image_directory, 'translate', path_trg[0])
        synths_src_path = os.path.join(image_directory, 'translate', path_src[0].replace('.png', '_A_' + dataset_letters[i] + '.png'))
        synths_trg_path = os.path.join(image_directory, 'translate', path_trg[0].replace('.png', '_' + dataset_letters[i] + '_A.png'))

        np_images_src = images_src.detach().cpu().numpy().squeeze()
        np_images_trg = images_trg.detach().cpu().numpy().squeeze()
        np_synths_src = synths_src.detach().cpu().numpy().squeeze()
        np_synths_trg = synths_trg.detach().cpu().numpy().squeeze()

        if not os.path.isfile(images_src_path):
            io.imsave(images_src_path, norm(np_images_src, config['input_dim'] != 1))
        if not os.path.isfile(images_trg_path):
            io.imsave(images_trg_path, norm(np_images_trg, config['input_dim'] != 1))
        if not os.path.isfile(synths_src_path):
            io.imsave(synths_src_path, norm(np_synths_src, config['input_dim'] != 1))
        if not os.path.isfile(synths_trg_path):
            io.imsave(synths_trg_path, norm(np_synths_trg, config['input_dim'] != 1))
"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, get_config, norm
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError:  # Will be 3.x series.
    pass
import gc
import os
import sys
import math
import shutil
import numpy as np
import time
import skimage

import pickle
import joblib

from skimage import io

from scipy import linalg

from sklearn import mixture
from sklearn import manifold
from sklearn import decomposition
from sklearn.metrics import balanced_accuracy_score
# from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import matplotlib as mpl

# python train.py --config configs/classification_MUNIT_None_0.0.yaml --snapshot_dir outputs/classification_MUNIT_None_0.0.yaml/checkpoints/ --resume 200

# Parsing input arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='configs/CXR_lungs', help='Path to the config file.')
parser.add_argument('--output_path', type=str,
                    default='.', help="Outputs path.")
parser.add_argument('--resume', type=int, default=-1)
parser.add_argument('--snapshot_dir', type=str, default='.')
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting.
config = get_config(opts.config)

# Setup model and data loader.
if config['trainer'] == 'MUNIT':
    trainer = MUNIT_Trainer(
        config, resume_epoch=opts.resume, snapshot_dir=opts.snapshot_dir)
elif config['trainer'] == 'UNIT':
    trainer = UNIT_Trainer(config, resume_epoch=opts.resume,
                           snapshot_dir=opts.snapshot_dir)
else:
    sys.exit("Only support MUNIT|UNIT.")
    os.exit()

trainer.cuda()

# Reading parameters from config file.
dataset_numbers = config['dataset_numbers']

samples = list()
dataset_probs = list()
augmentation = list()
for i in range(config['n_datasets']):
    samples.append(config['sample_' + dataset_numbers[i]])
    dataset_probs.append(config['prob_' + dataset_numbers[i]])
    augmentation.append(config['transform_' + dataset_numbers[i]])

# Normalizing probabilities.
dataset_probs = np.asarray(dataset_probs, dtype=np.float32)
dataset_probs = dataset_probs / dataset_probs.sum()

# Setting Dataloaders.
train_loader_list, test_loader_list = get_all_data_loaders(
    config, config['n_datasets'], samples, augmentation, config['trim'])

loader_sizes = list()

for l in train_loader_list:

    loader_sizes.append(len(l))

loader_sizes = np.asarray(loader_sizes)
n_batches = loader_sizes.min()

# Setup logger and output folders.
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
# Copy config file to output folder.
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

# log dir
if not os.path.exists('logs'):
    print("Creating directory: {}".format(image_directory))
    os.makedirs('logs')
# Start training.
epochs = config['max_epoch']

#time_epochs = list()
# total_jacc = list()
total_acc = list()
total_pred = list()
sup_loss_list = list()
dis_loss_list = list()
gen_loss_list = list()

for ep in range(max(opts.resume, 0), epochs):

    beg_time = time.time()
    print('Start of epoch ' + str(ep + 1) + '...')

    # In case configs have changes, load again at each epoch.
    config = get_config(opts.config)
    # Copy config file to output folder.
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

    # Updating learning rate for epoch.
    trainer.update_learning_rate()

    print('    Training...')
    for it, data in enumerate(zip(*train_loader_list)):

        x_list = list()
        y_list = list()
        use_list = list()

        for i in range(config['n_datasets']):

            x = data[i][0]
            y = data[i][1]
            use = data[i][2].to(dtype=torch.bool)  # uint8)

            x_list.append(x)
            y_list.append(y)
            use_list.append(use)

        # Randomly selecting datasets.
        perm = np.random.choice(
            config['n_datasets'], 2, replace=False, p=dataset_probs)
        print('        Ep: ' + str(ep + 1) + ', it: ' + str(it + 1) +
              '/' + str(n_batches) + ', domain pair: ' + str(perm))

        index_a = perm[0]
        index_b = perm[1]

        x_a = x_list[index_a]
        x_b = x_list[index_b]

        y_a = y_list[index_a]
        y_b = y_list[index_b]

        use_a = use_list[index_a]
        use_b = use_list[index_b]

        x_a, x_b = Variable(x_a.cuda()), Variable(x_b.cuda())

        # Main training code.
        dis_loss = None
        gen_loss = None
        sup_loss = None

        if (ep + 1) <= int(0.75 * epochs):

            # If in Full Training mode.
            trainer.set_sup_trainable(True)
            trainer.set_gen_trainable(True)

            dis_loss = trainer.dis_update(x_a, x_b, index_a, index_b, config)
            gen_loss = trainer.gen_update(x_a, x_b, index_a, index_b, config)

        else:

            # If in Supervision Tuning mode.
            trainer.set_sup_trainable(True)
            trainer.set_gen_trainable(False)

        # y_a = y_a.to(dtype=torch.long)
        # y_a[y_a > 0] = 1
        y_a = Variable(y_a.cuda(), requires_grad=False)

        # y_b = y_b.to(dtype=torch.long)
        # y_b[y_b > 0] = 1
        y_b = Variable(y_b.cuda(), requires_grad=False)

        if (ep + 1) <= int(0.25 * epochs):

            sup_loss = trainer.sup_update(
                x_a, x_b, y_a, y_b, index_a, index_b, use_a, use_b, config)

        else:

            if config['transfer_type'] == 'none':
                sup_loss = trainer.sup_update(
                    x_a, x_b, y_a, y_b, index_a, index_b, use_a, use_b, config)
            elif config['transfer_type'] == 'pseudo':
                sup_loss = trainer.pseudo_update(
                    x_a, x_b, y_a, y_b, index_a, index_b, use_a, use_b, config)
            elif config['transfer_type'] == 'mmd':
                sup_loss = trainer.mmd_intra_update(
                    x_a, x_b, y_a, y_b, index_a, index_b, use_a, use_b, config)
            elif config['transfer_type'] == 'mmd_inter':
                sup_loss = trainer.mmd_inter_update(
                    x_a, x_b, y_a, y_b, index_a, index_b, use_a, use_b, config)
            elif config['transfer_type'] == 'coral':
                sup_loss = trainer.coral_intra_update(
                    x_a, x_b, y_a, y_b, index_a, index_b, use_a, use_b, config)
            elif config['transfer_type'] == 'coral_inter':
                sup_loss = trainer.coral_inter_update(
                    x_a, x_b, y_a, y_b, index_a, index_b, use_a, use_b, config)
            else:
                print('Transfer Method not recognized: ' +
                      config['transfer_type'])
                exit(0)

        # Printing losses.
        loss_file = open('logs/' + opts.config.split('/')
                         [-1].replace('.yaml', '_loss.log'), 'a')

        if dis_loss is not None and gen_loss is not None:

            # loss_file.write(' % ())
            dis_loss_list.append(dis_loss)
            gen_loss_list.append(gen_loss)
            if sup_loss is not None:
                print('            S Loss: %.2f, D Loss: %.2f, G Loss: %.2f' % (
                    sup_loss.cpu().item(), dis_loss.cpu().item(), gen_loss.cpu().item()))
                loss_file.write('Ep: %d, It: %d, S Loss: %.2f, D Loss: %.2f, G Loss: %.2f\n' % (
                    ep + 1, it, sup_loss.cpu().item(), dis_loss.cpu().item(), gen_loss.cpu().item()))
                sup_loss_list.append(sup_loss)
            else:
                print('            S Loss: _____, D loss: %.2f, G Loss: %.2f' %
                      (dis_loss.cpu().item(), gen_loss.cpu().item()))
                loss_file.write('Ep: %d, It: %d, S Loss: _____, D Loss: %.2f, G Loss: %.2f\n' % (
                    ep + 1, it, dis_loss.cpu().item(), gen_loss.cpu().item()))
                sup_loss_list.append(-1.0)

        else:

            dis_loss_list.append(-1.0)
            gen_loss_list.append(-1.0)
            if sup_loss is not None:
                print('            S Loss: %.2f' % (sup_loss.cpu().item()))
                loss_file.write('Ep: %d, It: %d, S Loss: %.2f, D Loss: _____, G Loss: _____\n' % (
                    ep + 1, it, sup_loss.cpu().item()))
                sup_loss_list.append(sup_loss)
            else:
                print('            S Loss: _____')
                loss_file.write(
                    'Ep: %d, It: %d, S Loss: _____, D Loss: _____, G Loss: _____\n' % (ep + 1, it))
                sup_loss_list.append(-1.0)

        loss_file.close()

    end_time = time.time()

    dif_time = end_time - beg_time

    print('    Epoch %d duration: %.2f' % ((ep + 1), dif_time))
    time_file = open('logs/' + opts.config.split('/')
                     [-1].replace('.yaml', '_time.log'), 'a')
    time_file.write('Epoch %d duration: %.2f\n' % ((ep + 1), dif_time))
    time_file.close()

    if (ep + 1) % config['snapshot_save_epoch'] == 0:

        trainer.save(checkpoint_directory, (ep + 1))

    if (ep + 1) % config['snapshot_test_epoch'] == 0:

        acc_file = open('logs/' + opts.config.split('/')
                        [-1].replace('.yaml', '_acc.log'), 'a')

        epoch_acc = list()
        lab_list = list()
        iso_list = list()
        for i in range(config['n_datasets']):

            print('    Testing ' + dataset_numbers[i] + '...')

            dataset_labels = list()
            dataset_preds = list()
            for it, data in enumerate(test_loader_list[i]):

                x = data[0]
                y = data[1]
                use = data[2]
                path = data[3]

                x = Variable(x.cuda())

                y = y.to(dtype=torch.long)
                # y[y > 0] = 1
                y = Variable(y.cuda(), requires_grad=False)

                pred, prob, iso = trainer.sup_forward(x, y, i, config)

                dataset_preds.extend(pred.cpu().numpy())
                dataset_labels.extend(y.cpu().numpy())

                iso_list.append(iso.cpu().detach().numpy().squeeze())
                lab_list.append(i)

                # x_path = os.path.join(image_directory, 'originals', path[0])
                # y_path = os.path.join(image_directory, 'labels', path[0])
                # p_path = os.path.join(image_directory, 'predictions', path[0])
                # pr_path = os.path.join(image_directory, 'probability', path[0])

                np_x = x.cpu().numpy().squeeze()
                np_y = y.cpu().numpy().squeeze()
                np_pr = prob.detach().cpu().numpy().squeeze()

                # io.imsave(x_path, norm(np_x, config['input_dim'] != 1))
                # io.imsave(y_path, skimage.img_as_ubyte(norm(np_y)))
                # io.imsave(p_path, skimage.img_as_ubyte(norm(p)))
                # io.imsave(pr_path, np_pr)

                del x, y, use, path, np_x, np_y, iso, pred, prob, np_pr

                gc.collect()

            dataset_preds = np.asarray(dataset_preds)
            dataset_labels = np.asarray(dataset_labels)

            # cls_weights = compute_class_weight('balanced', np.unique(dataset_labels),
            #                                    dataset_labels)

            # print(cls_weights)
            # cls_weight_dict = {i: cls_weights[i]
            #                    for i in range(len(np.unique(dataset_labels)))}

            # sample_weights = compute_sample_weight(
            #     cls_weight_dict, dataset_labels)

            # print(sample_weights)
            # weight_class = np.unique(np.array(dataset_labels), return_counts=True)[
            #     1] / len(dataset_labels)

            # print(weight_class)

            # samples_weights = weight_class[dataset_labels]

            weighted_acc = balanced_accuracy_score(
                dataset_labels, dataset_preds, sample_weight=None)

            print('        Test ' + dataset_numbers[i] + ' Accuracy epoch ' + str(
                ep + 1) + ': ' + str(100 * weighted_acc))  # + ' +/- ' + str(100 * dataset_jacc.std()))

            acc_file.write('        Test ' + dataset_numbers[i] + ' Accuracy epoch ' + str(
                ep + 1) + ': ' + str(100 * weighted_acc) + '\n')  # + ' +/- ' + str(100 * dataset_jacc.std()) + '\n')

            epoch_acc.append(100 * weighted_acc)

        acc_file.close()

        total_acc.append(np.asarray(epoch_acc))

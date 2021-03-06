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
from sklearn.metrics import balanced_accuracy_score, precision_score, confusion_matrix, f1_score, recall_score
# from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import matplotlib as mpl
import traceback
import resource

# python train.py --config configs/classification_MUNIT_None_0.0.yaml --snapshot_dir outputs/classification_MUNIT_None_0.0.yaml/checkpoints/ --resume 200


def mem():
    print('Memory usage         : % 2.2f MB' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0, 1), file=open("logs/output_mem.txt", "a")
    )


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
train_loader_list, test_loader_list, train_samples_weights_list, test_samples_weights_list, total_class_count = get_all_data_loaders(
    config, config['n_datasets'], samples, augmentation, config['trim'])

# pesos_loss = sum(total_class_count) / \
#     (len(total_class_count) * total_class_count)
max_sample = max(total_class_count)
pesos_loss = [max_sample/x for x in total_class_count]
pesos_loss = torch.Tensor(pesos_loss)
print(pesos_loss)
# Setup model and data loader.
if config['trainer'] == 'MUNIT':
    trainer = MUNIT_Trainer(
        config, resume_epoch=opts.resume, snapshot_dir=opts.snapshot_dir, pesos_loss=pesos_loss)
elif config['trainer'] == 'UNIT':
    trainer = UNIT_Trainer(config, resume_epoch=opts.resume,
                           snapshot_dir=opts.snapshot_dir)
else:
    sys.exit("Only support MUNIT|UNIT.")
    os.exit()

trainer.cuda()


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
# total_acc = list()
total_pred = list()
sup_loss_list = list()
dis_loss_list = list()
gen_loss_list = list()

try:
    for ep in range(max(opts.resume, 0), epochs):

        beg_time = time.time()
        print('Start of epoch ' + str(ep + 1) + '...')

        # In case configs have changes, load again at each epoch.
        config = get_config(opts.config)
        # Copy config file to output folder.
        shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

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

                dis_loss = trainer.dis_update(
                    x_a, x_b, index_a, index_b, config)
                gen_loss = trainer.gen_update(
                    x_a, x_b, index_a, index_b, config)

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
                dis_loss_list.append(dis_loss.item())
                gen_loss_list.append(gen_loss.item())
                if sup_loss is not None:
                    print('            S Loss: %.2f, D Loss: %.2f, G Loss: %.2f' % (
                        sup_loss.cpu().item(), dis_loss.cpu().item(), gen_loss.cpu().item()))
                    loss_file.write('Ep: %d, It: %d, S Loss: %.2f, D Loss: %.2f, G Loss: %.2f\n' % (
                        ep + 1, it, sup_loss.cpu().item(), dis_loss.cpu().item(), gen_loss.cpu().item()))
                    sup_loss_list.append(sup_loss.item())
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
                    sup_loss_list.append(sup_loss.item())
                else:
                    print('            S Loss: _____')
                    loss_file.write(
                        'Ep: %d, It: %d, S Loss: _____, D Loss: _____, G Loss: _____\n' % (ep + 1, it))
                    sup_loss_list.append(-1.0)

            loss_file.close()

            # del dis_loss, gen_loss, sup_loss, x_a, x_b, y_a, y_b
            # gc.collect()

        end_time = time.time()

        mem()

        # Updating learning rate for epoch.
        trainer.update_learning_rate()

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

            # epoch_acc = list()
            # lab_list = list()
            # iso_list = list()
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

                    y = Variable(y.cuda(), requires_grad=False)

                    pred, prob, iso = trainer.sup_forward(x, y, i, config)

                    dataset_preds.extend(pred.cpu().numpy())
                    dataset_labels.extend(y.cpu().numpy())

                    # iso_list.append(iso.cpu().detach().numpy().squeeze())
                    # lab_list.append(i)

                    del x, y, use, path, iso, pred, prob

                    gc.collect()

                dataset_preds = np.asarray(dataset_preds)
                dataset_labels = np.asarray(dataset_labels)

                weighted_acc = balanced_accuracy_score(
                    dataset_labels, dataset_preds, sample_weight=test_samples_weights_list[i])

                precision = precision_score(
                    dataset_labels, dataset_preds, average='macro')

                cm = confusion_matrix(
                    dataset_labels, dataset_preds, normalize='true')

                f1 = f1_score(dataset_labels, dataset_preds, average='macro')

                recall = recall_score(
                    dataset_labels, dataset_preds, average='macro')

                print('        Test ' + dataset_numbers[i] + ' Balanced Accuracy epoch ' + str(
                    ep + 1) + ': ' + str(100 * weighted_acc))
                acc_file.write('        Test ' + dataset_numbers[i] + ' Balanced Accuracy epoch ' + str(
                    ep + 1) + ': ' + str(100 * weighted_acc) + '\n')

                print('        Test ' + dataset_numbers[i] + ' Precision epoch ' + str(
                    ep + 1) + ': ' + str(100 * precision))
                acc_file.write('        Test ' + dataset_numbers[i] + ' Precision epoch ' + str(
                    ep + 1) + ': ' + str(100 * precision) + '\n')

                print('        Test ' + dataset_numbers[i] + ' F1 Score epoch ' + str(
                    ep + 1) + ': ' + str(100 * f1))
                acc_file.write('        Test ' + dataset_numbers[i] + ' F1 Score epoch ' + str(
                    ep + 1) + ': ' + str(100 * f1) + '\n')

                print('        Test ' + dataset_numbers[i] + ' Recall epoch ' + str(
                    ep + 1) + ': ' + str(100 * recall))
                acc_file.write('        Test ' + dataset_numbers[i] + ' Recall epoch ' + str(
                    ep + 1) + ': ' + str(100 * recall) + '\n')

                print('        Test ' + dataset_numbers[i] + ' Confusion Matrix epoch ' + str(
                    ep + 1) + ': \n' + str(cm))
                acc_file.write('        Test ' + dataset_numbers[i] + ' Confusion Matrix epoch ' + str(
                    ep + 1) + ': \n' + str(cm) + '\n\n')

                # epoch_acc.append(100 * weighted_acc)

            acc_file.close()

            # total_acc.append(np.asarray(epoch_acc))


except Exception as e:     # most generic exception you can catch
    logf = open("logs/error.log", "a")
    logf.write(str(e))
    traceback.print_exc(file=logf)

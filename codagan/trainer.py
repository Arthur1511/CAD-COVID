"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen, UNet, resnet50
from utils import weights_init, get_model_list, get_scheduler, norm, jaccard
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as functional
import os
import mmd
import coral

import numpy as np


class MUNIT_Trainer(nn.Module):

    def __init__(self, hyperparameters, resume_epoch=-1, snapshot_dir=None, pesos_loss=None):

        super(MUNIT_Trainer, self).__init__()

        lr = hyperparameters['lr']

        # Initiate the networks.
        # Auto-encoder for domain a.
        self.gen = AdaINGen(hyperparameters['input_dim'] + hyperparameters['n_datasets'],
                            hyperparameters['gen'], hyperparameters['n_datasets'])
        # Discriminator for domain a.
        self.dis = MsImageDis(
            hyperparameters['input_dim'] + hyperparameters['n_datasets'], hyperparameters['dis'])

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']
        self.beta1 = hyperparameters['beta1']
        self.beta2 = hyperparameters['beta2']
        self.weight_decay = hyperparameters['weight_decay']
        self.cross_entropy_w = hyperparameters['cross_entropy_w']
        # Initiating and loader pretrained UNet.
        # self.sup = UNet(input_channels=hyperparameters['input_dim'], num_classes=2).cuda()
        self.sup = resnet50(num_classes=4).cuda()

        # Fix the noise used in sampling.
        self.s_a = torch.randn(8, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(8, self.style_dim, 1, 1).cuda()

        # Setup the optimizers.
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']

        dis_params = list(self.dis.parameters())
        gen_params = list(self.gen.parameters()) + list(self.sup.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(self.beta1, self.beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(self.beta1, self.beta2), weight_decay=hyperparameters['weight_decay'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization.
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))

        # Presetting one hot encoding vectors.
        self.one_hot_img = torch.zeros(
            hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 256, 256).cuda()
        self.one_hot_c = torch.zeros(
            hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 64, 64).cuda()

        for i in range(hyperparameters['n_datasets']):
            self.one_hot_img[i, :, i, :, :].fill_(1)
            self.one_hot_c[i, :, i, :, :].fill_(1)

        if resume_epoch != -1:

            self.resume(snapshot_dir, resume_epoch, hyperparameters)

        self.pesos_loss = pesos_loss

    def recon_criterion(self, input, target):

        return torch.mean(torch.abs(input - target))

    def semi_criterion(self, input, target):

        if self.pesos_loss is not None:

            class_weights = torch.FloatTensor(self.pesos_loss).cuda()
            loss = nn.CrossEntropyLoss(
                weight=class_weights, reduction='mean', ignore_index=-1).cuda()

        else:

            loss = nn.CrossEntropyLoss(
                reduction='mean', ignore_index=-1).cuda()

        return loss(input, target)

    def set_gen_trainable(self, train_bool):

        if train_bool:
            self.gen.train()
            for param in self.gen.parameters():
                param.requires_grad = True

        else:
            self.gen.eval()
            for param in self.gen.parameters():
                param.requires_grad = True

    def set_sup_trainable(self, train_bool):

        if train_bool:
            self.sup.train()
            for param in self.sup.parameters():
                param.requires_grad = True
        else:
            self.sup.eval()
            for param in self.sup.parameters():
                param.requires_grad = True

    def sup_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        c_a, s_a_prime = self.gen.encode(one_hot_x_a)
        c_b, s_b_prime = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_c_a = torch.cat([c_a, self.one_hot_c[d_index_a]], 1)
        one_hot_c_b = torch.cat([c_b, self.one_hot_c[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_c_a, s_a_prime)
        x_b_recon = self.gen.decode(one_hot_c_b, s_b_prime)

        # Decode (cross domain).
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]], 1)
        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        c_b_recon, s_a_recon = self.gen.encode(one_hot_x_ba)
        c_a_recon, s_b_recon = self.gen.encode(one_hot_x_ab)

        # Forwarding through supervised model.
        loss_semi_a = None
        loss_semi_b = None
        self.loss_sup_total = None

        # Computing supervised loss for dataset a.
        has_a_label = (use_a.sum().item() > 0)
        if has_a_label:
            # p_a = self.sup(c_a, use_a, True)
            # p_a_recon = self.sup(c_a_recon, use_a, True)
            p_a = self.sup(c_a)
            p_a_recon = self.sup(c_a_recon)
            loss_semi_a = self.semi_criterion(p_a, y_a) + \
                self.semi_criterion(p_a_recon, y_a)

        # Computing supervised loss for dataset b.
        has_b_label = (use_b.sum().item() > 0)
        if has_b_label:
            # p_b = self.sup(c_b, use_b, True)
            # p_b_recon = self.sup(c_b_recon, use_b, True)
            p_b = self.sup(c_b)
            p_b_recon = self.sup(c_b_recon)
            loss_semi_b = self.semi_criterion(p_b, y_b) + \
                self.semi_criterion(p_b_recon, y_b)

        # Computing final loss.
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_sup_total = loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_sup_total = loss_semi_a
        elif loss_semi_b is not None:
            self.loss_sup_total = loss_semi_b

        if self.loss_sup_total is not None:
            self.loss_sup_total.backward()
            self.gen_opt.step()

        return self.loss_sup_total

    def pseudo_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        c_a, s_a_prime = self.gen.encode(one_hot_x_a)
        c_b, s_b_prime = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_c_a = torch.cat([c_a, self.one_hot_c[d_index_a]], 1)
        one_hot_c_b = torch.cat([c_b, self.one_hot_c[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_c_a, s_a_prime)
        x_b_recon = self.gen.decode(one_hot_c_b, s_b_prime)

        # Decode (cross domain).
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]], 1)
        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        c_b_recon, s_a_recon = self.gen.encode(one_hot_x_ba)
        c_a_recon, s_b_recon = self.gen.encode(one_hot_x_ab)

        # Forwarding through supervised model.
        loss_semi_a = None
        loss_semi_b = None
        self.loss_sup_total = None

        # Computing pseudo loss for dataset a.
        p_a = self.sup(c_a, torch.full_like(use_a, 1), True)
        p_a_recon = self.sup(c_a_recon, torch.full_like(use_a, 1), True)
        loss_semi_a = self.semi_criterion(p_a_recon, p_a.data.max(1)[1])

        # Computing pseudo loss for dataset b.
        p_b = self.sup(c_b, torch.full_like(use_b, 1), True)
        p_b_recon = self.sup(c_b_recon, torch.full_like(use_b, 1), True)
        loss_semi_b = self.semi_criterion(p_b_recon, p_b.data.max(1)[1])

        # Computing supervised loss for dataset a.
        has_a_label = (use_a.sum().item() > 0)
        if has_a_label:
            p_a = self.sup(c_a, use_a, True)
            p_a_recon = self.sup(c_a_recon, use_a, True)
            loss_semi_a += self.semi_criterion(p_a, y_a[use_a, :, :]) + \
                self.semi_criterion(p_a_recon, y_a[use_a, :, :])

        # Computing supervised loss for dataset b.
        has_b_label = (use_b.sum().item() > 0)
        if has_b_label:
            p_b = self.sup(c_b, use_b, True)
            p_b_recon = self.sup(c_b_recon, use_b, True)
            loss_semi_b += self.semi_criterion(p_b, y_b[use_b, :, :]) + \
                self.semi_criterion(p_b_recon, y_b[use_b, :, :])

        # Computing final loss.
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_sup_total = loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_sup_total = loss_semi_a
        elif loss_semi_b is not None:
            self.loss_sup_total = loss_semi_b

        if self.loss_sup_total is not None:
            self.loss_sup_total.backward()
            self.gen_opt.step()

        return self.loss_sup_total

    def mmd_iso_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        c_a, s_a_prime = self.gen.encode(one_hot_x_a)
        c_b, s_b_prime = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_c_a = torch.cat([c_a, self.one_hot_c[d_index_a]], 1)
        one_hot_c_b = torch.cat([c_b, self.one_hot_c[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_c_a, s_a_prime)
        x_b_recon = self.gen.decode(one_hot_c_b, s_b_prime)

        # Decode (cross domain).
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]], 1)
        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        c_b_recon, s_a_recon = self.gen.encode(one_hot_x_ba)
        c_a_recon, s_b_recon = self.gen.encode(one_hot_x_ab)

        # Forwarding through supervised model.
        loss_semi_a = None
        loss_semi_b = None
        self.loss_sup_total = None

        # Forwarding samples from a through supervised model.
        p_a, fv_a = self.sup(c_a, torch.full_like(use_a, 1), False)
        p_a_recon, fv_a_recon = self.sup(
            c_a_recon, torch.full_like(use_a, 1), False)

        # Forwarding samples from b through supervised model.
        p_b, fv_b = self.sup(c_b, torch.full_like(use_b, 1), False)
        p_b_recon, fv_b_recon = self.sup(
            c_b_recon, torch.full_like(use_b, 1), False)

        # Linearizing feature maps for original samples.
        avg_c_a = functional.avg_pool2d(c_a, kernel_size=64).squeeze()
        avg_c_b = functional.avg_pool2d(c_b, kernel_size=64).squeeze()

        # Linearizing feature maps for reconstructed samples.
        avg_c_a_recon = functional.avg_pool2d(
            c_a_recon, kernel_size=64).squeeze()
        avg_c_b_recon = functional.avg_pool2d(
            c_b_recon, kernel_size=64).squeeze()

        # Computing MMD for isomorphic representations of datasets a and b.
        loss_semi_ab = mmd.mmd_rbf_accelerate(avg_c_a, avg_c_b)
        loss_semi_ab_recon = mmd.mmd_rbf_accelerate(
            avg_c_a_recon, avg_c_b_recon)

        # Computing supervised loss for dataset a.
        has_a_label = (use_a.sum().item() > 0)
        if has_a_label:
            p_a = self.sup(c_a, use_a, True)
            p_a_recon = self.sup(c_a_recon, use_a, True)
            loss_semi_a = self.semi_criterion(p_a, y_a[use_a, :, :]) + \
                self.semi_criterion(p_a_recon, y_a[use_a, :, :])

        # Computing supervised loss for dataset b.
        has_b_label = (use_b.sum().item() > 0)
        if has_b_label:
            p_b = self.sup(c_b, use_b, True)
            p_b_recon = self.sup(c_b_recon, use_b, True)
            loss_semi_b = self.semi_criterion(p_b, y_b[use_b, :, :]) + \
                self.semi_criterion(p_b_recon, y_b[use_b, :, :])

        # Computing final loss.
        self.loss_sup_total = loss_semi_ab + loss_semi_ab_recon
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_sup_total += loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_sup_total += loss_semi_a
        elif loss_semi_b is not None:
            self.loss_sup_total += loss_semi_b

        if self.loss_sup_total is not None:
            self.loss_sup_total.backward()
            self.gen_opt.step()

        return self.loss_sup_total

    def mmd_intra_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        c_a, s_a_prime = self.gen.encode(one_hot_x_a)
        c_b, s_b_prime = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_c_a = torch.cat([c_a, self.one_hot_c[d_index_a]], 1)
        one_hot_c_b = torch.cat([c_b, self.one_hot_c[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_c_a, s_a_prime)
        x_b_recon = self.gen.decode(one_hot_c_b, s_b_prime)

        # Decode (cross domain).
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]], 1)
        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        c_b_recon, s_a_recon = self.gen.encode(one_hot_x_ba)
        c_a_recon, s_b_recon = self.gen.encode(one_hot_x_ab)

        # Forwarding through supervised model.
        loss_semi_a = None
        loss_semi_b = None
        self.loss_sup_total = None

        # Forwarding samples from a through supervised model.
        p_a, fv_a = self.sup(c_a, torch.full_like(use_a, 1), False)
        p_a_recon, fv_a_recon = self.sup(
            c_a_recon, torch.full_like(use_a, 1), False)

        # Forwarding samples from b through supervised model.
        p_b, fv_b = self.sup(c_b, torch.full_like(use_b, 1), False)
        p_b_recon, fv_b_recon = self.sup(
            c_b_recon, torch.full_like(use_b, 1), False)

        # Linearizing feature maps for samples from dataset a.
        avg_a_4 = functional.avg_pool2d(fv_a[0], kernel_size=64).squeeze()
        avg_a_3 = functional.avg_pool2d(fv_a[1], kernel_size=128).squeeze()
        avg_a_2 = functional.avg_pool2d(fv_a[2], kernel_size=256).squeeze()
        avg_a_1 = functional.avg_pool2d(fv_a[3], kernel_size=256).squeeze()

        avg_a_recon_4 = functional.avg_pool2d(
            fv_a_recon[0], kernel_size=64).squeeze()
        avg_a_recon_3 = functional.avg_pool2d(
            fv_a_recon[1], kernel_size=128).squeeze()
        avg_a_recon_2 = functional.avg_pool2d(
            fv_a_recon[2], kernel_size=256).squeeze()
        avg_a_recon_1 = functional.avg_pool2d(
            fv_a_recon[3], kernel_size=256).squeeze()

        # Linearizing feature maps for samples from dataset b.
        avg_b_4 = functional.avg_pool2d(fv_b[0], kernel_size=64).squeeze()
        avg_b_3 = functional.avg_pool2d(fv_b[1], kernel_size=128).squeeze()
        avg_b_2 = functional.avg_pool2d(fv_b[2], kernel_size=256).squeeze()
        avg_b_1 = functional.avg_pool2d(fv_b[3], kernel_size=256).squeeze()

        avg_b_recon_4 = functional.avg_pool2d(
            fv_b_recon[0], kernel_size=64).squeeze()
        avg_b_recon_3 = functional.avg_pool2d(
            fv_b_recon[1], kernel_size=128).squeeze()
        avg_b_recon_2 = functional.avg_pool2d(
            fv_b_recon[2], kernel_size=256).squeeze()
        avg_b_recon_1 = functional.avg_pool2d(
            fv_b_recon[3], kernel_size=256).squeeze()

        # Computing MMD loss for dataset a.
        loss_semi_a = mmd.mmd_rbf_accelerate(avg_a_4, avg_a_recon_4) + \
            mmd.mmd_rbf_accelerate(avg_a_3, avg_a_recon_3) + \
            mmd.mmd_rbf_accelerate(avg_a_2, avg_a_recon_2) + \
            mmd.mmd_rbf_accelerate(avg_a_1, avg_a_recon_1)

        # Computing MMD loss for dataset b.
        loss_semi_b = mmd.mmd_rbf_accelerate(avg_b_4, avg_b_recon_4) + \
            mmd.mmd_rbf_accelerate(avg_b_3, avg_b_recon_3) + \
            mmd.mmd_rbf_accelerate(avg_b_2, avg_b_recon_2) + \
            mmd.mmd_rbf_accelerate(avg_b_1, avg_b_recon_1)

        # Computing supervised loss for dataset a.
        has_a_label = (use_a.sum().item() > 0)
        if has_a_label:
            p_a = self.sup(c_a, use_a, True)
            p_a_recon = self.sup(c_a_recon, use_a, True)
            loss_semi_a += self.semi_criterion(p_a, y_a[use_a, :, :]) + \
                self.semi_criterion(p_a_recon, y_a[use_a, :, :])

        # Computing supervised loss for dataset b.
        has_b_label = (use_b.sum().item() > 0)
        if has_b_label:
            p_b = self.sup(c_b, use_b, True)
            p_b_recon = self.sup(c_b_recon, use_b, True)
            loss_semi_b += self.semi_criterion(p_b, y_b[use_b, :, :]) + \
                self.semi_criterion(p_b_recon, y_b[use_b, :, :])

        # Computing final loss.
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_sup_total = loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_sup_total = loss_semi_a
        elif loss_semi_b is not None:
            self.loss_sup_total = loss_semi_b

        if self.loss_sup_total is not None:
            self.loss_sup_total.backward()
            self.gen_opt.step()

        return self.loss_sup_total

    def mmd_inter_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        c_a, s_a_prime = self.gen.encode(one_hot_x_a)
        c_b, s_b_prime = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_c_a = torch.cat([c_a, self.one_hot_c[d_index_a]], 1)
        one_hot_c_b = torch.cat([c_b, self.one_hot_c[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_c_a, s_a_prime)
        x_b_recon = self.gen.decode(one_hot_c_b, s_b_prime)

        # Decode (cross domain).
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]], 1)
        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        c_b_recon, s_a_recon = self.gen.encode(one_hot_x_ba)
        c_a_recon, s_b_recon = self.gen.encode(one_hot_x_ab)

        # Forwarding through supervised model.
        loss_semi_a = None
        loss_semi_b = None
        self.loss_sup_total = None

        # Forwarding samples from a through supervised model.
        p_a, fv_a = self.sup(c_a, torch.full_like(use_a, 1), False)
        p_a_recon, fv_a_recon = self.sup(
            c_a_recon, torch.full_like(use_a, 1), False)

        # Forwarding samples from b through supervised model.
        p_b, fv_b = self.sup(c_b, torch.full_like(use_b, 1), False)
        p_b_recon, fv_b_recon = self.sup(
            c_b_recon, torch.full_like(use_b, 1), False)

        # Linearizing feature maps for samples from dataset a.
        avg_a_4 = functional.avg_pool2d(fv_a[0], kernel_size=64).squeeze()
        avg_a_3 = functional.avg_pool2d(fv_a[1], kernel_size=128).squeeze()
        avg_a_2 = functional.avg_pool2d(fv_a[2], kernel_size=256).squeeze()
        avg_a_1 = functional.avg_pool2d(fv_a[3], kernel_size=256).squeeze()

        avg_a_recon_4 = functional.avg_pool2d(
            fv_a_recon[0], kernel_size=64).squeeze()
        avg_a_recon_3 = functional.avg_pool2d(
            fv_a_recon[1], kernel_size=128).squeeze()
        avg_a_recon_2 = functional.avg_pool2d(
            fv_a_recon[2], kernel_size=256).squeeze()
        avg_a_recon_1 = functional.avg_pool2d(
            fv_a_recon[3], kernel_size=256).squeeze()

        # Linearizing feature maps for samples from dataset b.
        avg_b_4 = functional.avg_pool2d(fv_b[0], kernel_size=64).squeeze()
        avg_b_3 = functional.avg_pool2d(fv_b[1], kernel_size=128).squeeze()
        avg_b_2 = functional.avg_pool2d(fv_b[2], kernel_size=256).squeeze()
        avg_b_1 = functional.avg_pool2d(fv_b[3], kernel_size=256).squeeze()

        avg_b_recon_4 = functional.avg_pool2d(
            fv_b_recon[0], kernel_size=64).squeeze()
        avg_b_recon_3 = functional.avg_pool2d(
            fv_b_recon[1], kernel_size=128).squeeze()
        avg_b_recon_2 = functional.avg_pool2d(
            fv_b_recon[2], kernel_size=256).squeeze()
        avg_b_recon_1 = functional.avg_pool2d(
            fv_b_recon[3], kernel_size=256).squeeze()

        # Computing MMD between datasets a and b original samples.
        loss_semi_ab = mmd.mmd_rbf_accelerate(avg_a_4, avg_b_4) + \
            mmd.mmd_rbf_accelerate(avg_a_3, avg_b_3) + \
            mmd.mmd_rbf_accelerate(avg_a_2, avg_b_2) + \
            mmd.mmd_rbf_accelerate(avg_a_1, avg_b_1)

        # Computing MMD between datasets a and b reconstructed samples.
        loss_semi_ab_recon = mmd.mmd_rbf_accelerate(avg_a_recon_4, avg_b_recon_4) + \
            mmd.mmd_rbf_accelerate(avg_a_recon_3, avg_b_recon_3) + \
            mmd.mmd_rbf_accelerate(avg_a_recon_2, avg_b_recon_2) + \
            mmd.mmd_rbf_accelerate(avg_a_recon_1, avg_b_recon_1)

        # Computing supervised loss for dataset a.
        has_a_label = (use_a.sum().item() > 0)
        if has_a_label:
            p_a = self.sup(c_a, use_a, True)
            p_a_recon = self.sup(c_a_recon, use_a, True)
            loss_semi_a = self.semi_criterion(p_a, y_a[use_a, :, :]) + \
                self.semi_criterion(p_a_recon, y_a[use_a, :, :])

        # Computing supervised loss for dataset b.
        has_b_label = (use_b.sum().item() > 0)
        if has_b_label:
            p_b = self.sup(c_b, use_b, True)
            p_b_recon = self.sup(c_b_recon, use_b, True)
            loss_semi_b = self.semi_criterion(p_b, y_b[use_b, :, :]) + \
                self.semi_criterion(p_b_recon, y_b[use_b, :, :])

        # Computing final loss.
        self.loss_sup_total = loss_semi_ab + loss_semi_ab_recon
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_sup_total += loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_sup_total += loss_semi_a
        elif loss_semi_b is not None:
            self.loss_sup_total += loss_semi_b

        if self.loss_sup_total is not None:
            self.loss_sup_total.backward()
            self.gen_opt.step()

        return self.loss_sup_total

    def coral_intra_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        c_a, s_a_prime = self.gen.encode(one_hot_x_a)
        c_b, s_b_prime = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_c_a = torch.cat([c_a, self.one_hot_c[d_index_a]], 1)
        one_hot_c_b = torch.cat([c_b, self.one_hot_c[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_c_a, s_a_prime)
        x_b_recon = self.gen.decode(one_hot_c_b, s_b_prime)

        # Decode (cross domain).
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]], 1)
        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        c_b_recon, s_a_recon = self.gen.encode(one_hot_x_ba)
        c_a_recon, s_b_recon = self.gen.encode(one_hot_x_ab)

        # Forwarding through supervised model.
        loss_semi_a = None
        loss_semi_b = None
        self.loss_sup_total = None

        # Forwarding samples from a through supervised model.
        p_a, fv_a = self.sup(c_a, torch.full_like(use_a, 1), False)
        p_a_recon, fv_a_recon = self.sup(
            c_a_recon, torch.full_like(use_a, 1), False)

        # Forwarding samples from b through supervised model.
        p_b, fv_b = self.sup(c_b, torch.full_like(use_b, 1), False)
        p_b_recon, fv_b_recon = self.sup(
            c_b_recon, torch.full_like(use_b, 1), False)

        # Linearizing feature maps for samples from dataset a.
        avg_a_4 = functional.avg_pool2d(fv_a[0], kernel_size=64).squeeze()
        avg_a_3 = functional.avg_pool2d(fv_a[1], kernel_size=128).squeeze()
        avg_a_2 = functional.avg_pool2d(fv_a[2], kernel_size=256).squeeze()
        avg_a_1 = functional.avg_pool2d(fv_a[3], kernel_size=256).squeeze()

        avg_a_recon_4 = functional.avg_pool2d(
            fv_a_recon[0], kernel_size=64).squeeze()
        avg_a_recon_3 = functional.avg_pool2d(
            fv_a_recon[1], kernel_size=128).squeeze()
        avg_a_recon_2 = functional.avg_pool2d(
            fv_a_recon[2], kernel_size=256).squeeze()
        avg_a_recon_1 = functional.avg_pool2d(
            fv_a_recon[3], kernel_size=256).squeeze()

        # Linearizing feature maps for samples from dataset b.
        avg_b_4 = functional.avg_pool2d(fv_b[0], kernel_size=64).squeeze()
        avg_b_3 = functional.avg_pool2d(fv_b[1], kernel_size=128).squeeze()
        avg_b_2 = functional.avg_pool2d(fv_b[2], kernel_size=256).squeeze()
        avg_b_1 = functional.avg_pool2d(fv_b[3], kernel_size=256).squeeze()

        avg_b_recon_4 = functional.avg_pool2d(
            fv_b_recon[0], kernel_size=64).squeeze()
        avg_b_recon_3 = functional.avg_pool2d(
            fv_b_recon[1], kernel_size=128).squeeze()
        avg_b_recon_2 = functional.avg_pool2d(
            fv_b_recon[2], kernel_size=256).squeeze()
        avg_b_recon_1 = functional.avg_pool2d(
            fv_b_recon[3], kernel_size=256).squeeze()

        # Computing CORAL for dataset a.
        loss_semi_a = coral.coral_loss(avg_a_4, avg_a_recon_4) + \
            coral.coral_loss(avg_a_3, avg_a_recon_3) + \
            coral.coral_loss(avg_a_2, avg_a_recon_2) + \
            coral.coral_loss(avg_a_1, avg_a_recon_1)

        # Computing CORAL for dataset b.
        loss_semi_b = coral.coral_loss(avg_b_4, avg_b_recon_4) + \
            coral.coral_loss(avg_b_3, avg_b_recon_3) + \
            coral.coral_loss(avg_b_2, avg_b_recon_2) + \
            coral.coral_loss(avg_b_1, avg_b_recon_1)

        # Computing supervised loss for dataset a.
        has_a_label = (use_a.sum().item() > 0)
        if has_a_label:
            p_a = self.sup(c_a, use_a, True)
            p_a_recon = self.sup(c_a_recon, use_a, True)
            loss_semi_a += self.semi_criterion(p_a, y_a[use_a, :, :]) + \
                self.semi_criterion(p_a_recon, y_a[use_a, :, :])

        # Computing supervised loss for dataset b.
        has_b_label = (use_b.sum().item() > 0)
        if has_b_label:
            p_b = self.sup(c_b, use_b, True)
            p_b_recon = self.sup(c_b_recon, use_b, True)
            loss_semi_b += self.semi_criterion(p_b, y_b[use_b, :, :]) + \
                self.semi_criterion(p_b_recon, y_b[use_b, :, :])

        # Computing final loss.
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_sup_total = loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_sup_total = loss_semi_a
        elif loss_semi_b is not None:
            self.loss_sup_total = loss_semi_b

        if self.loss_sup_total is not None:
            self.loss_sup_total.backward()
            self.gen_opt.step()

        return self.loss_sup_total

    def coral_inter_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        c_a, s_a_prime = self.gen.encode(one_hot_x_a)
        c_b, s_b_prime = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_c_a = torch.cat([c_a, self.one_hot_c[d_index_a]], 1)
        one_hot_c_b = torch.cat([c_b, self.one_hot_c[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_c_a, s_a_prime)
        x_b_recon = self.gen.decode(one_hot_c_b, s_b_prime)

        # Decode (cross domain).
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]], 1)
        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        c_b_recon, s_a_recon = self.gen.encode(one_hot_x_ba)
        c_a_recon, s_b_recon = self.gen.encode(one_hot_x_ab)

        # Forwarding through supervised model.
        loss_semi_a = None
        loss_semi_b = None
        self.loss_sup_total = None

        # Forwarding samples from a through supervised model.
        p_a, fv_a = self.sup(c_a, torch.full_like(use_a, 1), False)
        p_a_recon, fv_a_recon = self.sup(
            c_a_recon, torch.full_like(use_a, 1), False)

        # Forwarding samples from b through supervised model.
        p_b, fv_b = self.sup(c_b, torch.full_like(use_b, 1), False)
        p_b_recon, fv_b_recon = self.sup(
            c_b_recon, torch.full_like(use_b, 1), False)

        # Linearizing feature maps for samples from dataset a.
        avg_a_4 = functional.avg_pool2d(fv_a[0], kernel_size=64).squeeze()
        avg_a_3 = functional.avg_pool2d(fv_a[1], kernel_size=128).squeeze()
        avg_a_2 = functional.avg_pool2d(fv_a[2], kernel_size=256).squeeze()
        avg_a_1 = functional.avg_pool2d(fv_a[3], kernel_size=256).squeeze()

        avg_a_recon_4 = functional.avg_pool2d(
            fv_a_recon[0], kernel_size=64).squeeze()
        avg_a_recon_3 = functional.avg_pool2d(
            fv_a_recon[1], kernel_size=128).squeeze()
        avg_a_recon_2 = functional.avg_pool2d(
            fv_a_recon[2], kernel_size=256).squeeze()
        avg_a_recon_1 = functional.avg_pool2d(
            fv_a_recon[3], kernel_size=256).squeeze()

        # Linearizing feature maps for samples from dataset b.
        avg_b_4 = functional.avg_pool2d(fv_b[0], kernel_size=64).squeeze()
        avg_b_3 = functional.avg_pool2d(fv_b[1], kernel_size=128).squeeze()
        avg_b_2 = functional.avg_pool2d(fv_b[2], kernel_size=256).squeeze()
        avg_b_1 = functional.avg_pool2d(fv_b[3], kernel_size=256).squeeze()

        avg_b_recon_4 = functional.avg_pool2d(
            fv_b_recon[0], kernel_size=64).squeeze()
        avg_b_recon_3 = functional.avg_pool2d(
            fv_b_recon[1], kernel_size=128).squeeze()
        avg_b_recon_2 = functional.avg_pool2d(
            fv_b_recon[2], kernel_size=256).squeeze()
        avg_b_recon_1 = functional.avg_pool2d(
            fv_b_recon[3], kernel_size=256).squeeze()

        # Computing MMD between datasets a and b original samples.
        loss_semi_ab = coral.coral_loss(avg_a_4, avg_b_4) + \
            coral.coral_loss(avg_a_3, avg_b_3) + \
            coral.coral_loss(avg_a_2, avg_b_2) + \
            coral.coral_loss(avg_a_1, avg_b_1)

        # Computing MMD between datasets a and b reconstructed samples.
        loss_semi_ab_recon = coral.coral_loss(avg_a_recon_4, avg_b_recon_4) + \
            coral.coral_loss(avg_a_recon_3, avg_b_recon_3) + \
            coral.coral_loss(avg_a_recon_2, avg_b_recon_2) + \
            coral.coral_loss(avg_a_recon_1, avg_b_recon_1)

        # Computing supervised loss for dataset a.
        has_a_label = (use_a.sum().item() > 0)
        if has_a_label:
            p_a = self.sup(c_a, use_a, True)
            p_a_recon = self.sup(c_a_recon, use_a, True)
            loss_semi_a = self.semi_criterion(p_a, y_a[use_a, :, :]) + \
                self.semi_criterion(p_a_recon, y_a[use_a, :, :])

        # Computing supervised loss for dataset b.
        has_b_label = (use_b.sum().item() > 0)
        if has_b_label:
            p_b = self.sup(c_b, use_b, True)
            p_b_recon = self.sup(c_b_recon, use_b, True)
            loss_semi_b = self.semi_criterion(p_b, y_b[use_b, :, :]) + \
                self.semi_criterion(p_b_recon, y_b[use_b, :, :])

        # Computing final loss.
        self.loss_sup_total = loss_semi_ab + loss_semi_ab_recon
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_sup_total += loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_sup_total += loss_semi_a
        elif loss_semi_b is not None:
            self.loss_sup_total += loss_semi_b

        if self.loss_sup_total is not None:
            self.loss_sup_total.backward()
            self.gen_opt.step()

        return self.loss_sup_total

    def sup_forward(self, x, y, d_index, hyperparameters):

        self.sup.eval()

        # Encoding content image.
        one_hot_x = torch.cat(
            [x, self.one_hot_img[d_index, 0].unsqueeze(0)], 1)
        content, _ = self.gen.encode(one_hot_x)

        # Forwarding on supervised model.
        y_pred = self.sup(content)

        # Probabilities.
        prob = functional.softmax(y_pred, dim=1)

        # Computing metrics.
        # pred = y_pred.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        _, preds = torch.max(y_pred, 1)

        # jacc = jaccard(pred, y.cpu().squeeze(0).numpy())

        return preds, prob, content

    def translate(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.
        one_hot_x_a = torch.cat(
            [x_a, self.one_hot_img[d_index_a, 0].unsqueeze(0)], 1)
        one_hot_x_b = torch.cat(
            [x_b, self.one_hot_img[d_index_b, 0].unsqueeze(0)], 1)

        c_a, s_a_prime = self.gen.encode(one_hot_x_a)
        c_b, s_b_prime = self.gen.encode(one_hot_x_b)

        # Decode (cross domain).
        one_hot_c_ab = torch.cat(
            [c_a, self.one_hot_c[d_index_b, 0].unsqueeze(0)], 1)
        one_hot_c_ba = torch.cat(
            [c_b, self.one_hot_c[d_index_a, 0].unsqueeze(0)], 1)
#         x_ba = self.gen.decode(one_hot_c_ba, s_a)
#         x_ab = self.gen.decode(one_hot_c_ab, s_b)
        x_ba = self.gen.decode(one_hot_c_ba, s_a_prime)
        x_ab = self.gen.decode(one_hot_c_ab, s_b_prime)

        return x_ab, x_ba

    def gen_update(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        self.gen_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)

        c_a, s_a_prime = self.gen.encode(one_hot_x_a)
        c_b, s_b_prime = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_c_a = torch.cat([c_a, self.one_hot_c[d_index_a]], 1)
        one_hot_c_b = torch.cat([c_b, self.one_hot_c[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_c_a, s_a_prime)
        x_b_recon = self.gen.decode(one_hot_c_b, s_b_prime)

        # Decode (cross domain).
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]], 1)
        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        c_b_recon, s_a_recon = self.gen.encode(one_hot_x_ba)
        c_a_recon, s_b_recon = self.gen.encode(one_hot_x_ab)

        # Decode again (if needed).
        one_hot_c_aba_recon = torch.cat(
            [c_a_recon, self.one_hot_c[d_index_a]], 1)
        one_hot_c_bab_recon = torch.cat(
            [c_b_recon, self.one_hot_c[d_index_b]], 1)
        x_aba = self.gen.decode(one_hot_c_aba_recon, s_a_prime)
        x_bab = self.gen.decode(one_hot_c_bab_recon, s_b_prime)

        # Reconstruction loss.
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)

        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b)

        # GAN loss.
        self.loss_gen_adv_a = self.dis.calc_gen_loss(one_hot_x_ba)
        self.loss_gen_adv_b = self.dis.calc_gen_loss(one_hot_x_ab)

        # Total loss.
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
            hyperparameters['gan_w'] * self.loss_gen_adv_b + \
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
            hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
            hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
            hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
            hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
            hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
            hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b

        self.loss_gen_total.backward()
        self.gen_opt.step()

        return self.loss_gen_total

    def dis_update(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        self.dis_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        c_a, _ = self.gen.encode(one_hot_x_a)
        c_b, _ = self.gen.encode(one_hot_x_b)

        # Decode (cross domain).
        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]], 1)
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]], 1)
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # D loss.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        self.loss_dis_a = self.dis.calc_dis_loss(one_hot_x_ba, one_hot_x_a)
        self.loss_dis_b = self.dis.calc_dis_loss(one_hot_x_ab, one_hot_x_b)

        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + \
            hyperparameters['gan_w'] * self.loss_dis_b

        self.loss_dis_total.backward()
        self.dis_opt.step()

        return self.loss_dis_total

    def update_learning_rate(self):

        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, epoch, hyperparameters):

        print("--> " + checkpoint_dir)

        # Load generator.
        last_model_name = get_model_list(checkpoint_dir, epoch, "gen")
        print('Generative: ' + last_model_name)
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict)

        # Load discriminator.
        last_model_name = get_model_list(checkpoint_dir, epoch, "dis")
        print('Discriminator: ' + last_model_name)
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict)

        # Load supervised model.
        last_model_name = get_model_list(checkpoint_dir, epoch, "sup")
        print('Supervised: ' + last_model_name)
        state_dict = torch.load(last_model_name)
        self.sup.load_state_dict(state_dict)

        # Load optimizers.
        last_model_name = get_model_list(checkpoint_dir, epoch, "opt")
        print('Optimizers: ' + last_model_name)
        state_dict = torch.load(last_model_name)
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        for state in self.dis_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in self.gen_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        # Reinitilize schedulers.
        self.dis_scheduler = get_scheduler(
            self.dis_opt, hyperparameters, epoch)
        self.gen_scheduler = get_scheduler(
            self.gen_opt, hyperparameters, epoch)

        print('Resume from epoch %d' % epoch)
        return epoch

    def save(self, snapshot_dir, epoch):

        # Save generators, discriminators, and optimizers.
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % epoch)
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % epoch)
        sup_name = os.path.join(snapshot_dir, 'sup_%08d.pt' % epoch)
        opt_name = os.path.join(snapshot_dir, 'opt_%08d.pt' % epoch)

        torch.save(self.gen.state_dict(), gen_name)
        torch.save(self.dis.state_dict(), dis_name)
        torch.save(self.sup.state_dict(), sup_name)
        torch.save({'gen': self.gen_opt.state_dict(),
                    'dis': self.dis_opt.state_dict()}, opt_name)


class UNIT_Trainer(nn.Module):

    def __init__(self, hyperparameters, resume_epoch=-1, snapshot_dir=None):

        super(UNIT_Trainer, self).__init__()

        lr = hyperparameters['lr']

        # Initiate the networks.
        # Auto-encoder for domain a.
        self.gen = VAEGen(hyperparameters['input_dim'] + hyperparameters['n_datasets'],
                          hyperparameters['gen'], hyperparameters['n_datasets'])
        # Discriminator for domain a.
        self.dis = MsImageDis(
            hyperparameters['input_dim'] + hyperparameters['n_datasets'], hyperparameters['dis'])

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        self.sup = UNet(
            input_channels=hyperparameters['input_dim'], num_classes=2).cuda()

        # Setup the optimizers.
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']

        dis_params = list(self.dis.parameters())
        gen_params = list(self.gen.parameters()) + list(self.sup.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization.
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))

        # Presetting one hot encoding vectors.
        self.one_hot_img = torch.zeros(
            hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 256, 256).cuda()
        self.one_hot_h = torch.zeros(
            hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 64, 64).cuda()

        for i in range(hyperparameters['n_datasets']):
            self.one_hot_img[i, :, i, :, :].fill_(1)
            self.one_hot_h[i, :, i, :, :].fill_(1)

        if resume_epoch != -1:

            self.resume(snapshot_dir, resume_epoch, hyperparameters)

    def recon_criterion(self, input, target):

        return torch.mean(torch.abs(input - target))

    def semi_criterion(self, input, target):

        loss = nn.CrossEntropyLoss(reduction='mean').cuda()
        return loss(input, target)

    def __compute_kl(self, mu):

        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def set_gen_trainable(self, train_bool):

        if train_bool:
            self.gen.train()
            for param in self.gen.parameters():
                param.requires_grad = True

        else:
            self.gen.eval()
            for param in self.gen.parameters():
                param.requires_grad = True

    def set_sup_trainable(self, train_bool):

        if train_bool:
            self.sup.train()
            for param in self.sup.parameters():
                param.requires_grad = True
        else:
            self.sup.eval()
            for param in self.sup.parameters():
                param.requires_grad = True

    def sup_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        h_a, n_a = self.gen.encode(one_hot_x_a)
        h_b, n_b = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_h_a = torch.cat([h_a + n_a, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b = torch.cat([h_b + n_b, self.one_hot_h[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_h_a)
        x_b_recon = self.gen.decode(one_hot_h_b)

        # Decode (cross domain).
        one_hot_h_ab = torch.cat([h_a + n_a, self.one_hot_h[d_index_b]], 1)
        one_hot_h_ba = torch.cat([h_b + n_b, self.one_hot_h[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_h_ba)
        x_ab = self.gen.decode(one_hot_h_ab)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        h_b_recon, n_b_recon = self.gen.encode(one_hot_x_ba)
        h_a_recon, n_a_recon = self.gen.encode(one_hot_x_ab)

        # Decode again (if needed).
        one_hot_h_a_recon = torch.cat(
            [h_a_recon + n_a_recon, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b_recon = torch.cat(
            [h_b_recon + n_b_recon, self.one_hot_h[d_index_b]], 1)
        x_aba = self.gen.decode(
            one_hot_h_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen.decode(
            one_hot_h_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # Forwarding through supervised model.
        p_a = None
        p_b = None
        loss_semi_a = None
        loss_semi_b = None

        # Computing supervised loss for dataset a.
        has_a_label = (use_a.sum().item() > 0)
        if has_a_label:
            p_a = self.sup(h_a, use_a, True)
            p_a_recon = self.sup(h_a_recon, use_a, True)
            loss_semi_a = self.semi_criterion(p_a, y_a[use_a, :, :]) + \
                self.semi_criterion(p_a_recon, y_a[use_a, :, :])

        # Computing supervised loss for dataset b.
        has_b_label = (use_b.sum().item() > 0)
        if has_b_label:
            p_b = self.sup(h_b, use_b, True)
            p_b_recon = self.sup(h_b_recon, use_b, True)
            loss_semi_b = self.semi_criterion(p_b, y_b[use_b, :, :]) + \
                self.semi_criterion(p_b_recon, y_b[use_b, :, :])

        # Computing final loss.
        self.loss_sup_total = None
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_sup_total = loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_sup_total = loss_semi_a
        elif loss_semi_b is not None:
            self.loss_sup_total = loss_semi_b

        if self.loss_sup_total is not None:
            self.loss_sup_total.backward()
            self.gen_opt.step()

        return self.loss_sup_total

    def pseudo_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        h_a, n_a = self.gen.encode(one_hot_x_a)
        h_b, n_b = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_h_a = torch.cat([h_a + n_a, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b = torch.cat([h_b + n_b, self.one_hot_h[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_h_a)
        x_b_recon = self.gen.decode(one_hot_h_b)

        # Decode (cross domain).
        one_hot_h_ab = torch.cat([h_a + n_a, self.one_hot_h[d_index_b]], 1)
        one_hot_h_ba = torch.cat([h_b + n_b, self.one_hot_h[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_h_ba)
        x_ab = self.gen.decode(one_hot_h_ab)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        h_b_recon, n_b_recon = self.gen.encode(one_hot_x_ba)
        h_a_recon, n_a_recon = self.gen.encode(one_hot_x_ab)

        # Decode again (if needed).
        one_hot_h_a_recon = torch.cat(
            [h_a_recon + n_a_recon, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b_recon = torch.cat(
            [h_b_recon + n_b_recon, self.one_hot_h[d_index_b]], 1)
        x_aba = self.gen.decode(
            one_hot_h_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen.decode(
            one_hot_h_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # Forwarding through supervised model.
        p_a = None
        p_b = None
        loss_semi_a = None
        loss_semi_b = None

        # Computing pseudo loss for dataset a.
        p_a = self.sup(h_a, torch.full_like(use_a, 1), True)
        p_a_recon = self.sup(h_a_recon, torch.full_like(use_a, 1), True)
        loss_semi_a = self.semi_criterion(p_a_recon, p_a.data.max(1)[1])

        # Computing supervised loss for dataset a.
        has_a_label = (use_a.sum().item() > 0)
        if has_a_label:
            p_a = self.sup(h_a, use_a, True)
            p_a_recon = self.sup(h_a_recon, use_a, True)
            loss_semi_a += self.semi_criterion(p_a, y_a[use_a, :, :]) + \
                self.semi_criterion(p_a_recon, y_a[use_a, :, :])

        # Computing pseudo loss for dataset b.
        p_b = self.sup(h_b, torch.full_like(use_b, 1), True)
        p_b_recon = self.sup(h_b_recon, torch.full_like(use_b, 1), True)
        loss_semi_b = self.semi_criterion(p_b_recon, p_b.data.max(1)[1])

        # Computing supervised loss for dataset b.
        has_b_label = (use_b.sum().item() > 0)
        if has_b_label:
            p_b = self.sup(h_b, use_b, True)
            p_b_recon = self.sup(h_b_recon, use_b, True)
            loss_semi_b += self.semi_criterion(p_b, y_b[use_b, :, :]) + \
                self.semi_criterion(p_b_recon, y_b[use_b, :, :])

        # Computing final loss.
        self.loss_sup_total = None
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_sup_total = loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_sup_total = loss_semi_a
        elif loss_semi_b is not None:
            self.loss_sup_total = loss_semi_b

        if self.loss_sup_total is not None:
            self.loss_sup_total.backward()
            self.gen_opt.step()

        return self.loss_sup_total

    def mmd_intra_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        h_a, n_a = self.gen.encode(one_hot_x_a)
        h_b, n_b = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_h_a = torch.cat([h_a + n_a, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b = torch.cat([h_b + n_b, self.one_hot_h[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_h_a)
        x_b_recon = self.gen.decode(one_hot_h_b)

        # Decode (cross domain).
        one_hot_h_ab = torch.cat([h_a + n_a, self.one_hot_h[d_index_b]], 1)
        one_hot_h_ba = torch.cat([h_b + n_b, self.one_hot_h[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_h_ba)
        x_ab = self.gen.decode(one_hot_h_ab)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        h_b_recon, n_b_recon = self.gen.encode(one_hot_x_ba)
        h_a_recon, n_a_recon = self.gen.encode(one_hot_x_ab)

        # Decode again (if needed).
        one_hot_h_a_recon = torch.cat(
            [h_a_recon + n_a_recon, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b_recon = torch.cat(
            [h_b_recon + n_b_recon, self.one_hot_h[d_index_b]], 1)
        x_aba = self.gen.decode(
            one_hot_h_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen.decode(
            one_hot_h_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # Forwarding through supervised model.
        p_a = None
        p_b = None
        loss_semi_a = None
        loss_semi_b = None

        # Computing MMD for dataset a.
        p_a, fv_a = self.sup(h_a, torch.full_like(use_a, 1), False)
        p_a_recon, fv_a_recon = self.sup(
            h_a_recon, torch.full_like(use_a, 1), False)

        avg_a_4 = functional.avg_pool2d(fv_a[0], kernel_size=64).squeeze()
        avg_a_3 = functional.avg_pool2d(fv_a[1], kernel_size=128).squeeze()
        avg_a_2 = functional.avg_pool2d(fv_a[2], kernel_size=256).squeeze()
        avg_a_1 = functional.avg_pool2d(fv_a[3], kernel_size=256).squeeze()

        avg_a_recon_4 = functional.avg_pool2d(
            fv_a_recon[0], kernel_size=64).squeeze()
        avg_a_recon_3 = functional.avg_pool2d(
            fv_a_recon[1], kernel_size=128).squeeze()
        avg_a_recon_2 = functional.avg_pool2d(
            fv_a_recon[2], kernel_size=256).squeeze()
        avg_a_recon_1 = functional.avg_pool2d(
            fv_a_recon[3], kernel_size=256).squeeze()

        loss_semi_a = mmd.mmd_rbf_accelerate(avg_a_4, avg_a_recon_4) + \
            mmd.mmd_rbf_accelerate(avg_a_3, avg_a_recon_3) + \
            mmd.mmd_rbf_accelerate(avg_a_2, avg_a_recon_2) + \
            mmd.mmd_rbf_accelerate(avg_a_1, avg_a_recon_1)

        # Computing supervised loss for dataset a.
        has_a_label = (use_a.sum().item() > 0)
        if has_a_label:
            p_a = self.sup(h_a, use_a, True)
            p_a_recon = self.sup(h_a_recon, use_a, True)
            loss_semi_a += self.semi_criterion(p_a, y_a[use_a, :, :]) + \
                self.semi_criterion(p_a_recon, y_a[use_a, :, :])

        # Computing MMD for dataset b.
        p_b, fv_b = self.sup(h_b, torch.full_like(use_b, 1), False)
        p_b_recon, fv_b_recon = self.sup(
            h_b_recon, torch.full_like(use_b, 1), False)

        avg_b_4 = functional.avg_pool2d(fv_b[0], kernel_size=64).squeeze()
        avg_b_3 = functional.avg_pool2d(fv_b[1], kernel_size=128).squeeze()
        avg_b_2 = functional.avg_pool2d(fv_b[2], kernel_size=256).squeeze()
        avg_b_1 = functional.avg_pool2d(fv_b[3], kernel_size=256).squeeze()

        avg_b_recon_4 = functional.avg_pool2d(
            fv_b_recon[0], kernel_size=64).squeeze()
        avg_b_recon_3 = functional.avg_pool2d(
            fv_b_recon[1], kernel_size=128).squeeze()
        avg_b_recon_2 = functional.avg_pool2d(
            fv_b_recon[2], kernel_size=256).squeeze()
        avg_b_recon_1 = functional.avg_pool2d(
            fv_b_recon[3], kernel_size=256).squeeze()

        loss_semi_b = mmd.mmd_rbf_accelerate(avg_b_4, avg_b_recon_4) + \
            mmd.mmd_rbf_accelerate(avg_b_3, avg_b_recon_3) + \
            mmd.mmd_rbf_accelerate(avg_b_2, avg_b_recon_2) + \
            mmd.mmd_rbf_accelerate(avg_b_1, avg_b_recon_1)

        # Computing supervised loss for dataset b.
        has_b_label = (use_b.sum().item() > 0)
        if has_b_label:
            p_b = self.sup(h_b, use_b, True)
            p_b_recon = self.sup(h_b_recon, use_b, True)
            loss_semi_b += self.semi_criterion(p_b, y_b[use_b, :, :]) + \
                self.semi_criterion(p_b_recon, y_b[use_b, :, :])

        # Computing final loss.
        self.loss_sup_total = None
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_sup_total = loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_sup_total = loss_semi_a
        elif loss_semi_b is not None:
            self.loss_sup_total = loss_semi_b

        if self.loss_sup_total is not None:
            self.loss_sup_total.backward()
            self.gen_opt.step()

        return self.loss_sup_total

    def coral_intra_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        h_a, n_a = self.gen.encode(one_hot_x_a)
        h_b, n_b = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_h_a = torch.cat([h_a + n_a, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b = torch.cat([h_b + n_b, self.one_hot_h[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_h_a)
        x_b_recon = self.gen.decode(one_hot_h_b)

        # Decode (cross domain).
        one_hot_h_ab = torch.cat([h_a + n_a, self.one_hot_h[d_index_b]], 1)
        one_hot_h_ba = torch.cat([h_b + n_b, self.one_hot_h[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_h_ba)
        x_ab = self.gen.decode(one_hot_h_ab)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        h_b_recon, n_b_recon = self.gen.encode(one_hot_x_ba)
        h_a_recon, n_a_recon = self.gen.encode(one_hot_x_ab)

        # Decode again (if needed).
        one_hot_h_a_recon = torch.cat(
            [h_a_recon + n_a_recon, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b_recon = torch.cat(
            [h_b_recon + n_b_recon, self.one_hot_h[d_index_b]], 1)
        x_aba = self.gen.decode(
            one_hot_h_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen.decode(
            one_hot_h_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # Forwarding through supervised model.
        p_a = None
        p_b = None
        loss_semi_a = None
        loss_semi_b = None

        # Computing MMD for dataset a.
        p_a, fv_a = self.sup(h_a, torch.full_like(use_a, 1), False)
        p_a_recon, fv_a_recon = self.sup(
            h_a_recon, torch.full_like(use_a, 1), False)

        avg_a_4 = functional.avg_pool2d(fv_a[0], kernel_size=64).squeeze()
        avg_a_3 = functional.avg_pool2d(fv_a[1], kernel_size=128).squeeze()
        avg_a_2 = functional.avg_pool2d(fv_a[2], kernel_size=256).squeeze()
        avg_a_1 = functional.avg_pool2d(fv_a[3], kernel_size=256).squeeze()

        avg_a_recon_4 = functional.avg_pool2d(
            fv_a_recon[0], kernel_size=64).squeeze()
        avg_a_recon_3 = functional.avg_pool2d(
            fv_a_recon[1], kernel_size=128).squeeze()
        avg_a_recon_2 = functional.avg_pool2d(
            fv_a_recon[2], kernel_size=256).squeeze()
        avg_a_recon_1 = functional.avg_pool2d(
            fv_a_recon[3], kernel_size=256).squeeze()

        loss_semi_a = coral.coral_loss(avg_a_4, avg_a_recon_4) + \
            coral.coral_loss(avg_a_3, avg_a_recon_3) + \
            coral.coral_loss(avg_a_2, avg_a_recon_2) + \
            coral.coral_loss(avg_a_1, avg_a_recon_1)

        # Computing supervised loss for dataset a.
        has_a_label = (use_a.sum().item() > 0)
        if has_a_label:
            p_a = self.sup(h_a, use_a, True)
            p_a_recon = self.sup(h_a_recon, use_a, True)
            loss_semi_a += self.semi_criterion(p_a, y_a[use_a, :, :]) + \
                self.semi_criterion(p_a_recon, y_a[use_a, :, :])

        # Computing MMD for dataset b.
        p_b, fv_b = self.sup(h_b, torch.full_like(use_b, 1), False)
        p_b_recon, fv_b_recon = self.sup(h_b, torch.full_like(use_b, 1), False)

        avg_b_4 = functional.avg_pool2d(fv_b[0], kernel_size=64).squeeze()
        avg_b_3 = functional.avg_pool2d(fv_b[1], kernel_size=128).squeeze()
        avg_b_2 = functional.avg_pool2d(fv_b[2], kernel_size=256).squeeze()
        avg_b_1 = functional.avg_pool2d(fv_b[3], kernel_size=256).squeeze()

        avg_b_recon_4 = functional.avg_pool2d(
            fv_b_recon[0], kernel_size=64).squeeze()
        avg_b_recon_3 = functional.avg_pool2d(
            fv_b_recon[1], kernel_size=128).squeeze()
        avg_b_recon_2 = functional.avg_pool2d(
            fv_b_recon[2], kernel_size=256).squeeze()
        avg_b_recon_1 = functional.avg_pool2d(
            fv_b_recon[3], kernel_size=256).squeeze()

        loss_semi_b = coral.coral_loss(avg_b_4, avg_b_recon_4) + \
            coral.coral_loss(avg_b_3, avg_b_recon_3) + \
            coral.coral_loss(avg_b_2, avg_b_recon_2) + \
            coral.coral_loss(avg_b_1, avg_b_recon_1)

        # Computing supervised loss for dataset b.
        has_b_label = (use_b.sum().item() > 0)
        if has_b_label:
            p_b = self.sup(h_b, use_b, True)
            p_b_recon = self.sup(h_b_recon, use_b, True)
            loss_semi_b += self.semi_criterion(p_b, y_b[use_b, :, :]) + \
                self.semi_criterion(p_b_recon, y_b[use_b, :, :])

        # Computing final loss.
        self.loss_sup_total = None
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_sup_total = loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_sup_total = loss_semi_a
        elif loss_semi_b is not None:
            self.loss_sup_total = loss_semi_b

        if self.loss_sup_total is not None:
            self.loss_sup_total.backward()
            self.gen_opt.step()

        return self.loss_sup_total

    def sup_forward(self, x, y, d_index, hyperparameters):

        self.sup.eval()

        # Encoding content image.
        one_hot_x = torch.cat(
            [x, self.one_hot_img[d_index, 0].unsqueeze(0)], 1)
        hidden, _ = self.gen.encode(one_hot_x)

        # Forwarding on supervised model.
        y_pred = self.sup(hidden, only_prediction=True)

        # Probabilities.
        prob = functional.softmax(y_pred, dim=1)[:, 1]

        # Computing metrics.
        pred = y_pred.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        jacc = jaccard(pred, y.cpu().squeeze(0).numpy())

        return jacc, pred, prob, hidden

    def translate(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        # Encode.
        one_hot_x_a = torch.cat(
            [x_a, self.one_hot_img[d_index_a, 0].unsqueeze(0)], 1)
        one_hot_x_b = torch.cat(
            [x_b, self.one_hot_img[d_index_b, 0].unsqueeze(0)], 1)
        h_a, n_a = self.gen.encode(one_hot_x_a)
        h_b, n_b = self.gen.encode(one_hot_x_b)

        # Decode (cross domain).
        one_hot_h_ab = torch.cat(
            [h_a + n_a, self.one_hot_h[d_index_b, 0].unsqueeze(0)], 1)
        one_hot_h_ba = torch.cat(
            [h_b + n_b, self.one_hot_h[d_index_a, 0].unsqueeze(0)], 1)
        x_ba = self.gen.decode(one_hot_h_ba)
        x_ab = self.gen.decode(one_hot_h_ab)

        return x_ab, x_ba

    def gen_update(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        self.gen_opt.zero_grad()

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        h_a, n_a = self.gen.encode(one_hot_x_a)
        h_b, n_b = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_h_a = torch.cat([h_a + n_a, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b = torch.cat([h_b + n_b, self.one_hot_h[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_h_a)
        x_b_recon = self.gen.decode(one_hot_h_b)

        # Decode (cross domain).
        one_hot_h_ab = torch.cat([h_a + n_a, self.one_hot_h[d_index_b]], 1)
        one_hot_h_ba = torch.cat([h_b + n_b, self.one_hot_h[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_h_ba)
        x_ab = self.gen.decode(one_hot_h_ab)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        h_b_recon, n_b_recon = self.gen.encode(one_hot_x_ba)
        h_a_recon, n_a_recon = self.gen.encode(one_hot_x_ab)

        # Decode again (if needed).
        one_hot_h_a_recon = torch.cat(
            [h_a_recon + n_a_recon, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b_recon = torch.cat(
            [h_b_recon + n_b_recon, self.one_hot_h[d_index_b]], 1)
        x_aba = self.gen.decode(
            one_hot_h_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen.decode(
            one_hot_h_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # Reconstruction loss.
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)

        # GAN loss.
        self.loss_gen_adv_a = self.dis.calc_gen_loss(one_hot_x_ba)
        self.loss_gen_adv_b = self.dis.calc_gen_loss(one_hot_x_ab)

        # Total loss.
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
            hyperparameters['gan_w'] * self.loss_gen_adv_b + \
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
            hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
            hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
            hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
            hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
            hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
            hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab

        self.loss_gen_total.backward()
        self.gen_opt.step()

        return self.loss_gen_total

    def dis_update(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        self.dis_opt.zero_grad()

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        h_a, n_a = self.gen.encode(one_hot_x_a)
        h_b, n_b = self.gen.encode(one_hot_x_b)

        # Decode (cross domain).
        one_hot_h_ab = torch.cat([h_a + n_a, self.one_hot_h[d_index_b]], 1)
        one_hot_h_ba = torch.cat([h_b + n_b, self.one_hot_h[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_h_ba)
        x_ab = self.gen.decode(one_hot_h_ab)

        # D loss.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        self.loss_dis_a = self.dis.calc_dis_loss(
            one_hot_x_ba, one_hot_x_a)  # .detach(), one_hot_x_a)
        self.loss_dis_b = self.dis.calc_dis_loss(
            one_hot_x_ab, one_hot_x_b)  # .detach(), one_hot_x_b)

        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + \
            hyperparameters['gan_w'] * self.loss_dis_b

        self.loss_dis_total.backward()
        self.dis_opt.step()

        return self.loss_dis_total

    def update_learning_rate(self):

        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, epoch, hyperparameters):

        print("--> " + checkpoint_dir)

        # Load generator.
        last_model_name = get_model_list(checkpoint_dir, epoch, "gen")
        print('Generative: ' + last_model_name)
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict)

        # Load discriminator.
        last_model_name = get_model_list(checkpoint_dir, epoch, "dis")
        print('Discriminator: ' + last_model_name)
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict)

        # Load supervised model.
        last_model_name = get_model_list(checkpoint_dir, epoch, "sup")
        print('Supervised: ' + last_model_name)
        state_dict = torch.load(last_model_name)
        self.sup.load_state_dict(state_dict)

        # Load optimizers.
        last_model_name = get_model_list(checkpoint_dir, epoch, "opt")
        print('Optimizers: ' + last_model_name)
        state_dict = torch.load(last_model_name)
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        for state in self.dis_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in self.gen_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        # Reinitilize schedulers.
        self.dis_scheduler = get_scheduler(
            self.dis_opt, hyperparameters, epoch)
        self.gen_scheduler = get_scheduler(
            self.gen_opt, hyperparameters, epoch)

        print('Resume from epoch %d' % epoch)
        return epoch

    def save(self, snapshot_dir, epoch):

        # Save generators, discriminators, and optimizers.
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % epoch)
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % epoch)
        sup_name = os.path.join(snapshot_dir, 'sup_%08d.pt' % epoch)
        opt_name = os.path.join(snapshot_dir, 'opt_%08d.pt' % epoch)

        torch.save(self.gen.state_dict(), gen_name)
        torch.save(self.dis.state_dict(), dis_name)
        torch.save(self.sup.state_dict(), sup_name)
        torch.save({'gen': self.gen_opt.state_dict(),
                    'dis': self.dis_opt.state_dict()}, opt_name)

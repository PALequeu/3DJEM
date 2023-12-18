# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import utils
from utils import Hamiltonian, ISEBSW

import torch as t, torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
#import ipdb
import models as models
import numpy as np
import wideresnet
import json
# Sampling
from tqdm import tqdm
from data import ModelNet40
import matplotlib.pyplot as plt

t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
im_sz = 32
n_ch = 3


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * t.matmul(src, dst.permute(0, 2, 1))
    dist += t.sum(src ** 2, -1).view(B, N, 1)
    dist += t.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    group_dist, group_idx = t.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_dist, group_idx


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)

# TODO Switch the model from 2D Resnet to 3D PointMLP
class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(F, self).__init__()
        self.f = models.__dict__['pointMLP']()
        self.energy_output = nn.Linear(256, 1)
        self.class_output = nn.Linear(256, 5)

        checkpoint = t.load("./check_points/best_checkpoint.pth")
        checkpoint = checkpoint['net']
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # remove `module.`
            if name == "classifier.8.weight":
                self.class_output.weight.data = v[:5]
            elif name == "classifier.8.bias":
                self.class_output.bias.data = v[:5]
            else:
                new_state_dict[name] = v

        # self.f.load_state_dict(new_state_dict, strict=False)
        # self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        # self.energy_output = nn.Linear(self.f.last_dim, 1)
        # self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()

# TODO Switch the model from 2D Resnet to 3D PointMLP
class CCF(F):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(CCF, self).__init__(depth, width, norm=norm, dropout_rate=dropout_rate, n_classes=5)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])


def cycle(loader):
    while True:
        for data in loader:
            yield data


def grad_norm(m):
    total_norm = 0
    for p in m.parameters():
        param_grad = p.grad
        if param_grad is not None:
            param_norm = param_grad.data.norm(2) ** 2
            total_norm += param_norm
    total_norm = total_norm ** (1. / 2)
    return total_norm.item()


def grad_vals(m):
    ps = []
    for p in m.parameters():
        if p.grad is not None:
            ps.append(p.grad.data.view(-1))
    ps = t.cat(ps)
    return ps.mean().item(), ps.std(), ps.abs().mean(), ps.abs().std(), ps.abs().min(), ps.abs().max()


# TODO Change input dim from [bs, 3, H, W] to [bs, 3, num_points] (or [bs, num_points, 3])
# TODO Also, initialize them to uniformly lie on a sphere surface 
# TODO Consider how to uniformly distribute them without singularity
# TODO Also, find an appropriate radius of the sphere.
def init_random(args, bs):
    # return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)
    random_points = t.randn(bs * 1024, 3)
    random_points = random_points / random_points.norm(dim=-1, keepdim=True)
    # print(random_points)
    # assert 0
    return random_points.view(bs, 1024, 3).permute(0, 2, 1)


def get_model_and_buffer(args, device, sample_q):
    model_cls = F if args.uncond else CCF
    f = model_cls(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=args.n_classes)
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    if args.load_path is None:
        # make replay buffer
        replay_buffer = init_random(args, args.buffer_size)
    else:
        print(f"loading model from {args.load_path}")
        ckpt_dict = t.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]

    f = f.to(device)
    return f, replay_buffer


# TODO Load 3D Point cloud dataset instead of 2D images (from pointMLP code)
# TODO This should return dataloader for train/val/testset.
def get_data(args):
    
    
    # if args.dataset == "svhn":
    #     transform_train = tr.Compose(
    #         [tr.Pad(4, padding_mode="reflect"),
    #          tr.RandomCrop(im_sz),
    #          tr.ToTensor(),
    #          tr.Normalize((.5, .5, .5), (.5, .5, .5)),
    #          lambda x: x + args.sigma * t.randn_like(x)]
    #     )
    # else:
    #     transform_train = tr.Compose(
    #         [tr.Pad(4, padding_mode="reflect"),
    #          tr.RandomCrop(im_sz),
    #          tr.RandomHorizontalFlip(),
    #          tr.ToTensor(),
    #          tr.Normalize((.5, .5, .5), (.5, .5, .5)),
    #          lambda x: x + args.sigma * t.randn_like(x)]
    #     )
    # transform_test = tr.Compose(
    #     [tr.ToTensor(),
    #      tr.Normalize((.5, .5, .5), (.5, .5, .5)),
    #      lambda x: x + args.sigma * t.randn_like(x)]
    # )
    
    # def dataset_fn(train, transform):
    #     if args.dataset == "cifar10":
    #         return tv.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
    #     elif args.dataset == "cifar100":
    #         return tv.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
    #     else:
    #         return tv.datasets.SVHN(root=args.data_root, transform=transform, download=True,
    #                                 split="train" if train else "test")

    # # get all training inds
    # full_train = dataset_fn(True, transform_train)
    # all_inds = list(range(len(full_train)))
    # # set seed
    # np.random.seed(1234)
    # # shuffle
    # np.random.shuffle(all_inds)
    # # seperate out validation set
    # if args.n_valid is not None:
    #     valid_inds, train_inds = all_inds[:args.n_valid], all_inds[args.n_valid:]
    # else:
    #     valid_inds, train_inds = [], all_inds
    # train_inds = np.array(train_inds)
    # train_labeled_inds = []
    # other_inds = []
    # train_labels = np.array([full_train[ind][1] for ind in train_inds])
    # if args.labels_per_class > 0:
    #     for i in range(args.n_classes):
    #         print(i)
    #         train_labeled_inds.extend(train_inds[train_labels == i][:args.labels_per_class])
    #         other_inds.extend(train_inds[train_labels == i][args.labels_per_class:])
    # else:
    #     train_labeled_inds = train_inds
        
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=args.workers,
                             batch_size=args.batch_size // 2, shuffle=False, drop_last=False)

    # dset_train = DataSubset(
    #     dataset_fn(True, transform_train),
    #     inds=train_inds)
    # dset_train_labeled = DataSubset(
    #     dataset_fn(True, transform_train),
    #     inds=train_labeled_inds)
    # dset_valid = DataSubset(
    #     dataset_fn(True, transform_test),
    #     inds=valid_inds)
    
    # dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    # dload_train_labeled = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    # dload_train_labeled = cycle(dload_train_labeled)
    # dset_test = dataset_fn(False, transform_test)
    # dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
    # dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
    dload_train = train_loader
    dload_train_labeled = cycle(train_loader)
    dload_valid = test_loader
    dload_test = test_loader
    
    return dload_train, dload_train_labeled, dload_valid,dload_test


# TODO Modify [:, None, None, None] to [:, None, None]
def get_sample_q(args, device, dataset):
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random(args, bs), []
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes
        inds = t.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            assert not args.uncond, "Can't drawn conditional samples without giving me y"
        buffer_samples = replay_buffer[inds]
        random_samples = init_random(args, bs)
        random_samples.data = t.Tensor(dataset.data[inds, :args.num_points]).permute(0, 2, 1)
        choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None, None]
        # choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None, None, None]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.to(device), inds

    def sample_q(f, replay_buffer, y=None, n_steps=args.n_steps, in_steps= 10, args=args, save=True):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        f.eval()
        # get batch size
        bs = args.batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = t.autograd.Variable(init_sample, requires_grad=True)
       
        # sgld
            
        # # regularization for proximity
        if args.proximity:
            group_dist, _ = knn_point(10, x_k.permute(0, 2, 1), x_k.permute(0, 2, 1))
            reg_p = group_dist.sum() # should be mean rather than sum?      
            print(reg_p)
            for k in range(n_steps):
                group_dist, _ = knn_point(10, x_k.permute(0, 2, 1), x_k.permute(0, 2, 1))
                reg_p = group_dist.sum() # should be mean rather than sum?      
                print(reg_p)
                prox_grad = t.autograd.grad(group_dist.sum(), [x_k], retain_graph=False)[0] # retrain_graph should be True?
            
                f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
                x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k)

            x_k.data -= 0.1 * prox_grad

            group_dist, _ = knn_point(10, x_k.permute(0, 2, 1), x_k.permute(0, 2, 1))
            reg_p = group_dist.sum() # should be mean rather than sum?      
            print(reg_p)

        else:
            for k in range(n_steps):
                f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
                x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k)

        # if in_steps > 0:
        #     Hamiltonian_func = Hamiltonian(f.f.layer_one)

        # eps = args.eps
        # if args.pyld_lr <= 0:
        #     in_steps = 0

        # for it in range(n_steps):
        #     energies = f(x_k, y=y)
        #     e_x = energies.sum()
        #     # wgrad = f.f.conv1.weight.grad
        #     eta = t.autograd.grad(e_x, [x_k], retain_graph=True)[0]
        #     # e_x.backward(retain_graph=True)
        #     # eta = x_k.grad.detach()
        #     # f.f.conv1.weight.grad = wgrad
        #     # print("layer 1", f.f.layer_one)
        #     # print("layer 1 out", f.f.layer_one_out)
        #     if in_steps > 0:
        #         p = 1.0 * f.f.layer_one_out.grad
        #         p = p.detach()

        #     tmp_inp = x_k.data
        #     tmp_inp.requires_grad_()
        #     if args.sgld_lr > 0:
        #         # if in_steps == 0: use SGLD other than PYLD
        #         # if in_steps != 0: combine outter and inner gradients
        #         # default 0
        #         tmp_inp = x_k + t.clamp(eta, -eps, eps) * args.sgld_lr
        #         tmp_inp = t.clamp(tmp_inp, -1, 1)

        #     for i in range(in_steps):

        #         H = Hamiltonian_func(tmp_inp, p)

        #         eta_grad = t.autograd.grad(H, [tmp_inp], only_inputs=True, retain_graph=True)[0]
        #         eta_step = t.clamp(eta_grad, -eps, eps) * args.pyld_lr

        #         tmp_inp.data = tmp_inp.data + eta_step
        #         tmp_inp = t.clamp(tmp_inp, -1, 1)

        #     x_k.data = tmp_inp.data

        #     if args.sgld_std > 0.0:
        #         x_k.data += args.sgld_std * t.randn_like(x_k)

        # if in_steps > 0:
        #     loss = -1.0 * Hamiltonian_func(x_k.data, p)
        #     loss.backward()

        
        f.train()
        final_samples = x_k.detach()
        # update replay buffer
        # if len(replay_buffer) > 0 and save:
        #     replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples
    return sample_q

def sample_q(f, sample, y=None, n_steps=args.n_steps, in_steps= 10, args=args, save=True):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        f.eval()
        # get batch size
        # bs = args.batch_size if y is None else y.size(0)
        # # generate initial samples and buffer inds of those samples (if buffer is used)
        # init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = t.autograd.Variable(sample, requires_grad=True)
       
        # sgld
            
        # # regularization for proximity
        # if args.proximity:
        #     group_dist, _ = knn_point(10, x_k.permute(0, 2, 1), x_k.permute(0, 2, 1))
        #     reg_p = group_dist.sum() # should be mean rather than sum?      
        #     print(reg_p)
        #     for k in range(n_steps):
        #         group_dist, _ = knn_point(10, x_k.permute(0, 2, 1), x_k.permute(0, 2, 1))
        #         reg_p = group_dist.sum() # should be mean rather than sum?      
        #         print(reg_p)
        #         prox_grad = t.autograd.grad(group_dist.sum(), [x_k], retain_graph=False)[0] # retrain_graph should be True?
            
        #         f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
        #         x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k)

        #     x_k.data -= 0.1 * prox_grad

        #     group_dist, _ = knn_point(10, x_k.permute(0, 2, 1), x_k.permute(0, 2, 1))
        #     reg_p = group_dist.sum() # should be mean rather than sum?      
        #     print(reg_p)

        else:
            for k in range(n_steps):
                f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
                x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k)

        
        f.train()
        final_samples = x_k.detach()
        
        return final_samples

def eval_classification(f, dload, device):
    corrects, losses = [], []
    
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        x_p_d = x_p_d.permute(0, 2, 1)
        logits = f.classify(x_p_d)
        y_p_d = y_p_d.flatten()
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
        #print("loss test", ISEBSW(x_p_d, y_p_d, L=10, p=2, device="cpu"))
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss


def checkpoint(f, buffer, tag, args, device):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "replay_buffer": buffer
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)


def main(args):
    utils.makedirs(args.save_dir)
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    # datasets
    dload_train, dload_train_labeled, dload_valid, dload_test = get_data(args)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    dataset = ModelNet40(partition='train', num_points=args.num_points)
    sample_q = get_sample_q(args, device, dataset)
    f, replay_buffer = get_model_and_buffer(args, device, sample_q)
    print(replay_buffer.size())
    replay_buffer.data = t.Tensor(dataset.data[:args.buffer_size, :args.num_points]).permute(0, 2, 1)
    replay_buffer.data += t.randn_like(replay_buffer.data) * 0.025 # start from noisy dataset
    
    class MyDataParallel(t.nn.DataParallel):
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
    # f = MyDataParallel(f)

    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    # TODO how to plot 3d point clouds?
    # plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))
    #plot = lambda f_name, x: (plt.axes(projection='3d').scatter3D(x[0][0].cpu(), x[0][1].cpu(), x[0][2].cpu(), c=x[0][2].cpu(), cmap='Greens'), plt.savefig(f_name), plt.clf())
    # plot = lambda f_name, x: (fig = plt.figure, ax := plt.axes(projection='3d'),ax.scatter3D(x[0][0].cpu(), x[0][1].cpu(), x[0][2].cpu(), c=x[0][2].cpu(), cmap='Greens'), ax.set_xlim(-3, 3), ax.set_ylim(-3, 3), ax.set_zlim(-3, 3), plt.savefig(f_name),fig.clf())
    def plot(f_name, x, point_size=10, opacity=0.3):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(x[0][0].cpu(), x[0][1].cpu(), x[0][2].cpu(), c=x[0][2].cpu(), cmap='Greens', s=point_size, alpha=opacity)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.savefig(f_name)
        # plt.clf()
        plt.close('all') # to entirely close opened window
        
    # optimizer
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
    else:
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

    best_valid_acc = 0.0
    cur_iter = 0
    for epoch in range(args.n_epochs):
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * args.decay_rate
                param_group['lr'] = new_lr
            print("Decaying lr to {}".format(new_lr))
        print(len(dload_train))
        for i, (x_p_d, _) in tqdm(enumerate(dload_train)):

            # if i == 1:
            #     break
            if cur_iter <= args.warmup_iters:
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            x_p_d = x_p_d.to(device)
            x_p_d = x_p_d.permute(0, 2, 1)
            x_lab, y_lab = dload_train_labeled.__next__()
            x_lab, y_lab = x_lab.to(device), y_lab.to(device)
            x_lab = x_lab.permute(0, 2, 1)

            L = 0.
            if args.p_x_weight > 0:  # maximize log p(x)
                if args.class_cond_p_x_sample:
                    assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                    y_q = t.randint(0, 5, (args.batch_size,)).to(device)
                    x_q = sample_q(f, replay_buffer, y=y_q)
                else:
                    x_q = sample_q(f, replay_buffer)  # sample from log-sumexp
                # print(i, x_p_d.size())

                fp_all = f(x_p_d)
                fq_all = f(x_q)
                fp = fp_all.mean()
                fq = fq_all.mean()

                l_p_x = -(fp - fq) # original
                # l_p_x = fp - fq
                if cur_iter % args.print_every == 0:
                    print('P(x) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                   fp - fq))
                L += args.p_x_weight * l_p_x

            if args.p_y_given_x_weight > 0:  # maximize log p(y | x)
                logits = f.classify(x_lab)
                y_lab = y_lab.flatten()
                # print(logits.size())
                # print(logits)
                l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)
                if cur_iter % args.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch,
                                                                                 cur_iter,
                                                                                 l_p_y_given_x.item(),
                                                                                 acc.item()))
                    
                
                L += args.p_y_given_x_weight * l_p_y_given_x

            if args.p_x_y_weight > 0:  # maximize log p(x, y)
                assert not args.uncond, "this objective can only be trained for class-conditional EBM DUUUUUUUUHHHH!!!"
                x_q_lab = sample_q(f, replay_buffer, y=y_lab)
                fp, fq = f(x_lab, y_lab).mean(), f(x_q_lab, y_lab).mean()
                l_p_x_y = -(fp - fq)
                if cur_iter % args.print_every == 0:
                    print('P(x, y) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                      fp - fq))

                L += args.p_x_y_weight * l_p_x_y

            # break if the loss diverged...easier for poppa to run experiments this way
            if L.abs().item() > 1e8:
                print("BAD BOIIIIIIIIII")
                1/0

            optim.zero_grad()
            L.backward()
            optim.step()
            cur_iter += 1

            if cur_iter % 10 == 0:
                if args.plot_uncond:
                    if args.class_cond_p_x_sample:
                        assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                        # y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                        y_q = t.randint(0, 5, (args.batch_size,)).to(device)
                        x_q = sample_q(f, replay_buffer, y=y_q)
                    else:
                        x_q = sample_q(f, replay_buffer)
                    plot('{}/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)
                if args.plot_cond:  # generate class-conditional samples
                    y = t.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                    x_q_y = sample_q(f, replay_buffer, y=y)
                    plot('{}/x_q_y{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q_y)

        if epoch % args.ckpt_every == 0:
            checkpoint(f, replay_buffer, f'ckpt_{epoch}.pt', args, device)

        if epoch % args.eval_every == 0 and (args.p_y_given_x_weight > 0 or args.p_x_y_weight > 0):
            f.eval()
            with t.no_grad():
                # validation set
                correct, loss = eval_classification(f, dload_valid, device)
                print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    checkpoint(f, replay_buffer, "best_valid_ckpt.pt", args, device)
                # test set
                correct, loss = eval_classification(f, dload_test, device)
                print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))
            f.train()
        checkpoint(f, replay_buffer, "last_ckpt.pt", args, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "cifar100"])
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting
    parser.add_argument("--p_x_weight", type=float, default=1.)
    parser.add_argument("--p_y_given_x_weight", type=float, default=1.)
    parser.add_argument("--p_x_y_weight", type=float, default=0.)
    # regularization
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2,
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"],
                        help="norm to add to weights, none works fine")
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=20,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true",
                        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                             "Sample quality higher if set, but classification accuracy better if not.")
    parser.add_argument("--buffer_size", type=int, default=1000)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=.01)
    parser.add_argument("--sgld_std", type=float, default=1e-4)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--ckpt_every", type=int, default=1, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=10, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--proximity", action="store_true", help="If true, use proximity loss")
    parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
    parser.add_argument("--plot_cond", action="store_true", help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
    parser.add_argument("--n_valid", type=int, default=5000)
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--workers', default=1, type=int, help='workers')
    parser.add_argument("--pyld_lr", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1, help="eps bound")
    parser.add_argument("--log_arg", type=str, default='JEMPP-n_steps-in_steps-pyld_lr-norm-plc')
    parser.add_argument("--in_steps", type=int, default=20, help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")

    

    args = parser.parse_args()
    args.n_classes = 100 if args.dataset == "cifar100" else 5
    main(args)
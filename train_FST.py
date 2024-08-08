from __future__ import print_function
import os
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from advertorch.attacks import LinfPGDAttack
import torch.utils.data as data_utils
from torch.utils.data import Dataset
from resnet import *

import time
import datetime
from sampler_large import CustomSampler
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 AT')

parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch_size')
parser.add_argument('--lambda_intra', type=float, default=12,
                    help='lambda_intra')
parser.add_argument('--lambda_inter', type=float, default=90,
                    help='lambda_inter')


parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./result',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--decay-rate', default=0.85 ,type=float)

args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, epoch):
    """decrease lr"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, model, device, train_loader, optimizer, epoch, model_nat, model_temp):
    cnt = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)

        # calculate robust loss
        model.eval()
        model_temp.eval()
        data_adv = craft_adv_train(model=model_temp, x_natural=data, y=label)

        model.train()
        model_temp.train()
        optimizer.zero_grad()

        loss_sim = fst_loss(model=model_temp, x_natural=data, x_adv=data_adv, y=label, cnt=cnt, model_nat=model_nat)
        cnt = cnt + 1

        logits_out = model(data_adv)
        loss_ce = F.cross_entropy(logits_out, label)

        loss = loss_ce + loss_sim

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_ce: {:.6f} Loss_sim: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss_ce.item(), loss_sim.item()))


def fst_loss(model, x_natural, x_adv, y, cnt, model_nat):
    data, data_adv, label = x_natural.to("cuda"), x_adv.to("cuda"), y.to("cuda")

    cls_list = label.unique().tolist()
    cls = {lab: [] for lab in cls_list}
    label_count = [0]
    for x, y in zip(data, label):
        cls[y.item()].append(torch.unsqueeze(x, 0))
    for i in cls_list:
        temp = torch.cat(cls[i])
        label_count.append(len(temp))

    index = [sum(label_count[:i + 1]) for i in range(len(label_count))]
    sorted_indices = sorted(range(len(data)), key=lambda i: label[i])

    a4,a3,feat_adv,a1 = model(data_adv)
    b4,b3,feat_nat,b1 = model_nat(data)
    norm_feat_adv = F.normalize(feat_adv[sorted_indices], dim=-1)
    norm_feat_nat = F.normalize(feat_nat[sorted_indices], dim=-1)

    mat_nat2nat = torch.mm(norm_feat_nat, norm_feat_nat.t())
    mat_adv2adv = torch.mm(norm_feat_adv, norm_feat_adv.t())

    loss_same = 0
    cnt_same = 0
    loss_diff = 0
    cnt_diff = 0
    mat_dis_1 = 1 - mat_adv2adv
    mat_dis_2 = mat_nat2nat - mat_adv2adv
    for i in range(len(index) - 1):
        mat_same = mat_dis_1[index[i]:index[i + 1], index[i]:index[i + 1]]
        loss_same = loss_same + torch.sum(torch.exp(torch.abs(mat_same)))
        cnt_same = cnt_same + len(mat_same) * len(mat_same)

        mat_diff = mat_dis_2[index[i]:index[i + 1], index[i + 1]:]
        loss_diff = loss_diff + torch.sum(torch.exp(torch.abs(mat_diff)))
        cnt_diff = cnt_diff + torch.numel(mat_diff)

    if cnt % 200 == 0:
        print("\nloss_same:{}, loss_diff:{}\n".format(loss_same/cnt_same, loss_diff/cnt_diff))

    return (loss_same/cnt_same) * args.lambda_intra + (loss_diff/cnt_diff) * args.lambda_inter

def craft_adv_train(model, x_natural, y):
    data, label = x_natural.to("cuda"), y.to("cuda")
    data_ori = data

    cls_list = label.unique().tolist()
    cls = {label: [] for label in cls_list}
    label_count = [0]
    for x, y in zip(data, label):
        cls[y.item()].append(torch.unsqueeze(x, 0))
    for i in cls_list:
        temp = torch.cat(cls[i])
        label_count.append(len(temp))
    index = [sum(label_count[:i + 1]) for i in range(len(label_count))]
    label_sort = torch.sort(label)[0]

    data_adv = data_ori
    data_adv.requires_grad = True
    sorted_indices = sorted(range(len(data)), key=lambda i: label[i])
    for _ in range(10):
        feat4, feat3, feat2, feat1 = model(data_adv)
        feat_adv = feat1
        true_probs = torch.gather(F.softmax(feat_adv[sorted_indices], dim=1), 1, (label_sort.unsqueeze(1)).long()).squeeze()
        true_probs_matrix = torch.mm(true_probs.unsqueeze(1), true_probs.unsqueeze(1).t())

        norm_feat_adv = F.normalize(feat3[sorted_indices], dim=-1)
        mat_diffcls_3 = torch.mm(norm_feat_adv, norm_feat_adv.t()) * (1 - true_probs_matrix)
        mat_samecls_3 = torch.mm(norm_feat_adv, norm_feat_adv.t()) * true_probs_matrix
        norm_feat_adv = F.normalize(feat4[sorted_indices], dim=-1)
        mat_diffcls_4 = torch.mm(norm_feat_adv, norm_feat_adv.t()) * (1 - true_probs_matrix)
        mat_samecls_4 = torch.mm(norm_feat_adv, norm_feat_adv.t()) * true_probs_matrix
        mat_diff = mat_diffcls_4 + mat_diffcls_3
        mat_same = mat_samecls_4 + mat_samecls_3

        loss = 0
        for i in range(len(index) - 1):
            loss = loss - torch.mean(mat_same[index[i]:index[i + 1], index[i]:index[i + 1]])

        for i in range(len(index) - 2):
            loss = loss + torch.mean(mat_diff[index[i]:index[i + 1], index[i + 1]:])

        loss = loss + F.cross_entropy(feat1, label)

        grad = torch.autograd.grad(loss, data_adv, retain_graph=True)[0]
        data_adv = data_adv + 2 / 255 * grad.sign()
        delta = torch.clamp(data_adv - data_ori, min=-8 / 255, max=8 / 255)
        data_adv = torch.clamp(data_ori + delta, min=0, max=1)

    return data_adv


def craft_adv_test(model, x_natural, y):
    attack = LinfPGDAttack(model, eps=8/255, nb_iter=10, eps_iter=2/255, targeted=False)
    x_adv = attack.perturb(x_natural, y)
    return x_adv


def eval_test(model, device, test_loader):
    model.eval()
    correct = 0
    correct_adv = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)

        logits_out = model(data)
        pred = logits_out.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()

        data = craft_adv_test(model=model, x_natural=data, y=label)

        logits_out = model(data)
        pred = logits_out.max(1, keepdim=True)[1]
        correct_adv += pred.eq(label.view_as(pred)).sum().item()

    print('Test: Accuracy: {}/{} ({:.5f}%), Robust Accuracy: {}/{} ({:.5f}%)'.format(
        correct,     len(test_loader.dataset), 100. * correct     / len(test_loader.dataset),
        correct_adv, len(test_loader.dataset), 100. * correct_adv / len(test_loader.dataset)))

def weight_average(model, new_model, decay_rate, init=False):
    model.eval()
    new_model.eval()
    state_dict = model.state_dict()
    new_dict = new_model.state_dict()
    if init:
        decay_rate = 0
    for key in state_dict:
        new_dict[key] = (state_dict[key]*decay_rate + new_dict[key]*(1-decay_rate)).clone().detach()
    model.load_state_dict(new_dict)

def main():
    # settings
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor()
    ])

    # setup data loader
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(os.path.expanduser('./data/'),
                                                                train=True,
                                                                download=True,
                                                                transform=trans_train),
                                               batch_size=args.batch_size, num_workers=4, shuffle=True,
                                               drop_last=False, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(os.path.expanduser('./data/'),
                                                               train=False,
                                                               download=True,
                                                               transform=trans_test),
                                              batch_size=args.batch_size, num_workers=4, shuffle=False,
                                              drop_last=False, pin_memory=True)

    # ============================== training for CIFAR-100 with the designed sampler =========================
    # data_cifar100_train = datasets.CIFAR100('./data/', train=True, download=True, transform=trans_train)
    # batch_sampler = CustomSampler(data_cifar100_train, batch_size=256, num_classes=64)
    # train_loader = torch.utils.data.DataLoader(data_cifar100_train, num_workers=4, batch_sampler=batch_sampler, pin_memory=True)
    # ============================== training for CIFAR-100 with the designed sampler =========================

    # init model
    model = ResNet18().to(device)
    model_temp = ResNet_temp(model).to(device)
    model_nat = ResNet_temp(model).to(device)

    model = torch.nn.DataParallel(model)
    model_temp = torch.nn.DataParallel(model_temp)

    EMA_model = ResNet18().to(device)
    EMA_model = torch.nn.DataParallel(EMA_model)
    EMA_model.eval()

    best_path = torch.load("./result/train_nat_cifar10/100_ori_res18.pth")
    model_nat.load_state_dict(best_path)
    model_nat = torch.nn.DataParallel(model_nat)
    model_nat.eval()

    cudnn.benchmark = True
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        print('time:', et)
        adjust_learning_rate(optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch, model_nat, model_temp)

        weight_average(EMA_model, model, args.decay_rate, epoch < 50)
        # save checkpoint
        if epoch >= 75:
            if epoch % args.save_freq == 0:
                torch.save(EMA_model.module.state_dict(),
                           os.path.join(model_dir, '{}_res18.pth'.format(epoch)))
                torch.save(model.module.state_dict(),
                           os.path.join(model_dir, '{}_ori_res18.pth'.format(epoch)))
                print('save the model')
        eval_test(EMA_model, device, test_loader)
        if epoch >= 50:
            eval_test(model, device, test_loader)
        print('================================================================')

if __name__ == '__main__':
    main()

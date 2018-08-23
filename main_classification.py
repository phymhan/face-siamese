import os
import argparse
import random
import functools
import math
import numpy as np
from scipy import stats
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.utils


###############################################################################
# Options | Argument Parser
###############################################################################
class Options():
    def initialize(self, parser):
        parser.add_argument('--mode', type=str, default='train', help='train | test | visualize')
        parser.add_argument('--name', type=str, default='exp', help='experiment name')
        parser.add_argument('--dataroot', required=True, default='datasets/UTKFace', help='path to images')
        parser.add_argument('--datafile', type=str, default='', help='text file listing images')
        parser.add_argument('--dataroot_val', type=str, default='')
        parser.add_argument('--datafile_val', type=str, default='')
        parser.add_argument('--pretrained_model_path', type=str, default='pretrained_models/alexnet.pth', help='path to pretrained models')
        parser.add_argument('--use_pretrained_model', action='store_true', help='use pretrained model')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints')
        parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
        parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=100, help='batch size')
        parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load')
        parser.add_argument('--which_model', type=str, default='alexnet', help='which model')
        parser.add_argument('--n_layers', type=int, default=3, help='only used if which_model==n_layers')
        parser.add_argument('--nf', type=int, default=64, help='# of filters in first conv layer')
        parser.add_argument('--pooling', type=str, default='', help='empty: no pooling layer, max: MaxPool, avg: AvgPool')
        parser.add_argument('--loadSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--age_bins', nargs='+', type=int, default=[1, 11, 21, 31, 41, 51, 61, 71, 81, 91], help='list of bins, the (i+1)-th group is in the range [age_binranges[i], age_binranges[i+1]), e.g. [1, 11, 21, ..., 101], the 1-st group is [1, 10], the 9-th [91, 100], however, the 10-th [101, +inf)')
        parser.add_argument('--weight', nargs='+', type=float, default=[], help='weights for CE')
        parser.add_argument('--dropout', type=float, default=0.5, help='dropout p')
        parser.add_argument('--finetune_fc_only', action='store_true', help='fix feature extraction weights and finetune fc layers only, if True')
        parser.add_argument('--fc_dim', type=int, nargs='+', default=[3, 3], help='dimension of fc')
        parser.add_argument('--fc_relu_slope', type=float, default=0.5)
        parser.add_argument('--print_freq', type=int, default=50, help='print loss every print_freq iterations')
        parser.add_argument('--display_id', type=int, default=1, help='visdom window id, to disable visdom set id = -1.')
        parser.add_argument('--display_port', type=int, default=8097)
        parser.add_argument('--transforms', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--affineScale', nargs='+', type=float, default=[0.95, 1.05], help='scale tuple in transforms.RandomAffine')
        parser.add_argument('--affineDegrees', type=float, default=5, help='range of degrees in transforms.RandomAffine')
        parser.add_argument('--use_color_jitter', action='store_true', help='if specified, add color jitter in transforms')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        return parser

    def get_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(parser)
        self.opt = self.parser.parse_args()
        self.opt.use_gpu = len(self.opt.gpu_ids) > 0 and torch.cuda.is_available()
        self.opt.isTrain = self.opt.mode == 'train'
        assert(self.opt.num_classes == len(self.opt.age_bins))
        # set age_bins
        self.opt.age_bins_with_inf = self.opt.age_bins + [float('inf')]
        # weight
        if self.opt.weight:
            assert(len(self.opt.weight) == self.opt.num_classes)
        self.print_options(self.opt)
        return self.opt
    
    def print_options(self, opt):
        message = ''
        message += '--------------- Options -----------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


###############################################################################
# Dataset and Dataloader
###############################################################################
class SiameseNetworkDataset(Dataset):
    def __init__(self, rootdir, source_file_path, transform=None):
        self.rootdir = rootdir
        self.source_file_path = source_file_path
        self.transform = transform
        with open(self.source_file_path, 'r') as f:
            self.source_file = f.readlines()

    def __getitem__(self, index):
        ss = self.source_file[index].split()
        imgA = Image.open(os.path.join(self.rootdir, ss[0])).convert('RGB')
        imgB = Image.open(os.path.join(self.rootdir, ss[1])).convert('RGB')
        label = int(ss[2])
        if self.transform is not None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB, torch.LongTensor(1).fill_(label)

    def __len__(self):
        return len(self.source_file)


class SingleImageDataset(Dataset):
    def __init__(self, rootdir, source_file, transform=None):
        self.rootdir = rootdir
        self.transform = transform
        if source_file:
            with open(source_file, 'r') as f:
                self.source_file = f.readlines()
        else:
            self.source_file = os.listdir(rootdir)

    def __getitem__(self, index):
        imgA = Image.open(os.path.join(self.rootdir, self.source_file[index].rstrip('\n'))).convert('RGB')
        if self.transform is not None:
            imgA = self.transform(imgA)
        return imgA, self.source_file[index]

    def __len__(self):
        return len(self.source_file)


###############################################################################
# Networks and Models
###############################################################################
# models: alexnet, alexnet+gap, nlayerclassifier
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    
    def load_pretrained(self, state_dict):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        if 'classifier.6.weight' in state_dict:
            del state_dict['classifier.6.weight']
        if 'classifier.6.bias' in state_dict:
            del state_dict['classifier.6.bias']
        self.load_state_dict(state_dict, strict=False)
    
    def get_finetune_parameters(self):
        # used when opt.finetune_fc_only is True
        return self.classifier.parameters()


# a lighter alexnet, with fewer params in fc layers
class AlexNetLite(nn.Module):
    def __init__(self, num_classes=10, pooling='avg', dropout=0.5):
        super(AlexNetLite, self).__init__()
        self.pooling = pooling
        fw = 1 if pooling else 6
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * fw * fw, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        if self.pooling == 'avg':
            x = nn.AvgPool2d(x.size(2))(x)
        elif self.pooling == 'max':
            x = nn.MaxPool2d(x.size(2))(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def load_pretrained(self, state_dict):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        for key in list(state_dict.keys()):
            if key.startswith('classifier'):
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=True)


class ResNet(nn.Module):
    def __init__(self, num_classes, which_model):
        super(ResNet, self).__init__()
        model = None
        if which_model == 'resnet18':
            from torchvision.models import resnet18
            model = resnet18(False)
            model.fc = nn.Linear(512 * 1, num_classes)
        elif which_model == 'resnet34':
            from torchvision.models import resnet34
            model = resnet34(False)
            model.fc = nn.Linear(512 * 1, num_classes)
        elif which_model == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(False)
            model.fc = nn.Linear(512 * 4, num_classes)
        elif which_model == 'resnet101':
            from torchvision.models import resnet101
            model = resnet101(False)
            model.fc = nn.Linear(512 * 4, num_classes)
        elif which_model == 'resnet152':
            from torchvision.models import resnet152
            model = resnet152(False)
            model.fc = nn.Linear(512 * 4, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)

    def load_pretrained(self, state_dict):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        for key in list(state_dict.keys()):
            if key.startswith('fc'):
                state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


# borrowed from NLayerDiscriminator in pix2pix-cyclegan
class NLayerClassifier(nn.Module):
    def __init__(self, num_classes=10, nf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerClassifier, self).__init__()
        input_nc = 3
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(nf * nf_mult_prev, nf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(nf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(nf * nf_mult_prev, nf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(nf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        self.features = nn.Sequential(*sequence)

        sequence = [
            nn.Conv2d(nf * nf_mult, num_classes, kernel_size=1, stride=1, padding=0),
        ]

        self.classifier = nn.Sequential(*sequence)

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        y = nn.AvgPool2d(x.size(2))(x)  # x.size(2)==x.size(3)
        y = self.classifier(y)
        return y.view(y.size(0), -1)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


###############################################################################
# Helper Functions | Utilities
###############################################################################
def imshow(img, text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_age_label(fname, binranges):
    strlist = fname.split('_')
    age = float(strlist[0])
    l = None
    for l in range(len(binranges)-1):
        if (age >= binranges[l]) and (age < binranges[l+1]):
            break
    return l


def get_prediction(score):
    batch_size = score.size(0)
    score_cpu = score.detach().cpu().numpy()
    pred = stats.mode(score_cpu.argmax(axis=1).reshape(batch_size, -1), axis=1)
    return pred[0].reshape(batch_size)


###############################################################################
# Main Routines
###############################################################################
def get_model(opt):
    # define model
    net = None
    if opt.which_model == 'alexnet':
        net = AlexNet(opt.num_classes)
    elif opt.which_model == 'alexnet_lite':
        net = AlexNetLite(opt.num_classes, opt.pooling, opt.dropout)
    elif 'resnet' in opt.which_model:
        net = ResNet(opt.num_classes, opt.which_model)
    elif opt.which_model == 'n_layers':
        net = NLayerClassifier(num_classes=opt.num_classes, nf=opt.nf, n_layers=opt.n_layers)
    else:
        raise NotImplementedError('Model [%s] is not implemented.' % opt.which_model)
    
    # initialize | load weights
    if opt.mode == 'train':
        net.apply(weights_init)
        if opt.use_pretrained_model:
            if isinstance(net, torch.nn.DataParallel):
                net.module.load_pretrained(opt.pretrained_model_path)
            else:
                net.load_pretrained(opt.pretrained_model_path)
    else:
        net.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, opt.name, '{}_net.pth'.format(opt.which_epoch))))
        net.eval()
    
    if opt.use_gpu:
        net.cuda()
    return net


def get_transform(opt):
    transform_list = []
    if opt.transforms == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.transforms == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.transforms == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.transforms == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.transforms == 'none':
        transform_list.append(transforms.Lambda(
            lambda img: __adjust(img)))
    elif opt.transforms == 'resize_affine_crop':
        transform_list.append(transforms.Resize([opt.loadSize, opt.loadSize], Image.BICUBIC))
        transform_list.append(transforms.RandomAffine(degrees=opt.affineDegrees, scale=tuple(opt.affineScale),
                                                      resample=Image.BICUBIC, fillcolor=127))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.transforms == 'resize_affine_center':
        transform_list.append(transforms.Resize([opt.loadSize, opt.loadSize], Image.BICUBIC))
        transform_list.append(transforms.RandomAffine(degrees=opt.affineDegrees, scale=tuple(opt.affineScale),
                                                      resample=Image.BICUBIC, fillcolor=127))
        transform_list.append(transforms.CenterCrop(opt.fineSize))
    else:
        raise ValueError('--resize_or_crop %s is not a valid option.' % opt.transforms)

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if opt.isTrain and opt.use_color_jitter:
        transform_list.append(transforms.ColorJitter())  # TODO

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    return transforms.Compose(transform_list)


# Routines for training
def train(opt, net, dataloader):
    if len(opt.weight):
        opt.weight = torch.Tensor(opt.weight)
        if opt.use_gpu:
            opt.weight = opt.weight.cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=opt.weight)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    opt.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    if opt.finetune_fc_only:
        if isinstance(net, nn.DataParallel):
            param = net.module.get_finetune_parameters()
        else:
            param = net.get_finetune_parameters()
    else:
        param = net.parameters()
    optimizer = optim.Adam(param, lr=opt.lr)

    dataset_size, dataset_size_val = opt.dataset_size, opt.dataset_size_val
    loss_history = []
    total_iter = 0
    num_iter_per_epoch = math.ceil(dataset_size / opt.batch_size)
    opt.display_val_acc = not not dataloader_val
    loss_legend = ['classification']
    if opt.display_id >= 0:
        import visdom
        vis = visdom.Visdom(server='http://localhost', port=opt.display_port)
        # plot_data = {'X': [], 'Y': [], 'leg': ['loss']}
        plot_loss = {'X': [], 'Y': [], 'leg': loss_legend}
        plot_acc = {'X': [], 'Y': [], 'leg': ['train', 'val'] if opt.display_val_acc else ['train']}

    for epoch in range(1, opt.num_epochs+1):
        epoch_iter = 0
        pred_train = []
        target_train = []

        for i, data in enumerate(dataloader, 0):
            img0, path0 = data
            label = [get_age_label(name, opt.age_bins_with_inf) for name in path0]
            label = torch.LongTensor(label)
            if opt.use_gpu:
                img0, label = img0.cuda(), label.cuda()
            epoch_iter += 1
            total_iter += 1

            optimizer.zero_grad()

            output = net.forward(img0)
            loss = criterion(output, label)

            # get predictions
            pred_train.append(get_prediction(output))
            target_train.append(label.cpu().numpy())

            loss.backward()
            optimizer.step()

            losses = {'classification': loss.item()}
            
            if total_iter % opt.print_freq == 0:
                print("epoch %02d, iter %06d, loss: %.4f" % (epoch, total_iter, loss.item()))
                if opt.display_id >= 0:
                    plot_loss['X'].append(epoch + epoch_iter/num_iter_per_epoch)
                    plot_loss['Y'].append([losses[k] for k in plot_loss['leg']])
                    vis.line(
                        X=np.stack([np.array(plot_loss['X'])] * len(plot_loss['leg']), 1),
                        Y=np.array(plot_loss['Y']),
                        opts={'title': 'loss', 'legend': plot_loss['leg'], 'xlabel': 'epoch', 'ylabel': 'loss'},
                        win=opt.display_id
                    )
                loss_history.append(loss.item())
        
        curr_acc = {}
        # evaluate training
        err_train = np.count_nonzero(np.concatenate(pred_train) - np.concatenate(target_train)) / dataset_size
        curr_acc['train'] = 1 - err_train

        # evaluate val
        if opt.display_val_acc:
            pred_val = []
            target_val = []
            for i, data in enumerate(dataloader_val, 0):
                img0, path0 = data
                label = [get_age_label(name, opt.age_bins_with_inf) for name in path0]
                label = torch.LongTensor(label)
                if opt.use_gpu:
                    img0 = img0.cuda()
                output = net.forward(img0)
                pred_val.append(get_prediction(output))
                target_val.append(label.cpu().numpy())
            err_val = np.count_nonzero(np.concatenate(pred_val) - np.concatenate(target_val)) / dataset_size_val
            curr_acc['val'] = 1 - err_val
        
        # plot accs
        if opt.display_id >= 0:
            plot_acc['X'].append(epoch)
            plot_acc['Y'].append([curr_acc[k] for k in plot_acc['leg']])
            vis.line(
                X=np.stack([np.array(plot_acc['X'])] * len(plot_acc['leg']), 1),
                Y=np.array(plot_acc['Y']),
                opts={'title': 'accuracy', 'legend': plot_acc['leg'], 'xlabel': 'epoch', 'ylabel': 'accuracy'},
                win=opt.display_id+1
            )
            sio.savemat(os.path.join(opt.save_dir, 'mat_loss'), plot_loss)
            sio.savemat(os.path.join(opt.save_dir, 'mat_acc'), plot_acc)

        torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, 'latest_net.pth'))
        if opt.use_gpu:
            net.cuda()
        if epoch % opt.save_epoch_freq == 0:
            torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, '{}_net.pth'.format(epoch)))
            if opt.use_gpu:
                net.cuda()

    with open(os.path.join(opt.save_dir, 'loss.txt'), 'w') as f:
        for loss in loss_history:
            f.write(str(loss)+'\n')
    # show_plot(counter, loss_history)


# Routines for testing
def test(opt, net, dataloader):
    pred = []
    target = []
    acc = 0
    for i, data in enumerate(dataloader, 0):
        img0, path0 = data
        label = [get_age_label(name, opt.age_bins_with_inf) for name in path0]
        label = torch.LongTensor(label)
        if opt.use_gpu:
            img0, label = img0.cuda(), label.cuda()
        output = net.forward(img0)
        target.append(label.cpu().detach().numpy().squeeze())
        pred.append(output.cpu().detach().numpy().squeeze().argmax())
        if target[-1] == pred[-1]:
            acc += 1
        print('--> image #%d: target %d   pred %d' % (i+1, target[-1], pred[-1]))

    print('================================================================================')
    print('accuracy: %.6f' % (100. * acc/len(pred)))


###############################################################################
# main()
###############################################################################
# TODO: set random seed

if __name__=='__main__':
    opt = Options().get_options()

    # get model
    net = get_model(opt)

    if opt.mode == 'train':
        # get dataloader
        dataset = SingleImageDataset(opt.dataroot, opt.datafile, transform=get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=True, num_workers=opt.num_workers, batch_size=opt.batch_size)
        opt.dataset_size = len(dataset)
        # val dataset
        if opt.dataroot_val:
            dataset_val = SingleImageDataset(opt.dataroot_val, opt.datafile_val, transform=get_transform(opt))
            dataloader_val = DataLoader(dataset_val, shuffle=True, num_workers=opt.num_workers, batch_size=opt.batch_size)
            opt.dataset_size_val = len(dataset_val)
        else:
            dataloader_val = None
            opt.dataset_size_val = 0
        print('dataset size = %d' % len(dataset))
        # train
        train(opt, net, dataloader)
    elif opt.mode == 'test':
        # get dataloader
        dataset = SingleImageDataset(opt.dataroot, opt.datafile, transform=get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=1, batch_size=1)
        opt.dataset_size_val = len(dataset)
        # test
        test(opt, net, dataloader)
    else:
        raise NotImplementedError('Mode [%s] is not implemented.' % opt.mode)

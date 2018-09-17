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
        parser.add_argument('--pretrained_model_path', type=str, default='', help='path to pretrained models')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints')
        parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
        parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=100, help='batch size')
        parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load')
        parser.add_argument('--which_model', type=str, default='resnet18', help='which model')
        parser.add_argument('--n_layers', type=int, default=3, help='only used if which_model==n_layers')
        parser.add_argument('--nf', type=int, default=64, help='# of filters in first conv layer')
        parser.add_argument('--pooling', type=str, default='avg', help='empty: no pooling layer, max: MaxPool, avg: AvgPool')
        parser.add_argument('--loadSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--age_bins', nargs='+', type=int, default=[1, 11, 21, 31, 41, 51, 61, 71, 81, 91], help='list of bins, the (i+1)-th group is in the range [age_binranges[i], age_binranges[i+1]), e.g. [1, 11, 21, ..., 101], the 1-st group is [1, 10], the 9-th [91, 100], however, the 10-th [101, +inf)')
        parser.add_argument('--weight', nargs='+', type=float, default=[], help='weights for CE')
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout p')
        parser.add_argument('--finetune_fc_only', action='store_true', help='fix feature extraction weights and finetune fc layers only, if True')
        parser.add_argument('--fc_dim', type=int, nargs='+', default=[3, 3], help='dimension of fc')
        parser.add_argument('--fc_relu_slope', type=float, default=0.5)
        parser.add_argument('--cnn_dim', type=int, nargs='+', default=[64, 1], help='cnn kernel dims for feature dimension reduction')
        parser.add_argument('--cnn_pad', type=int, default=1, help='padding of cnn layers defined by cnn_dim')
        parser.add_argument('--cnn_relu_slope', type=float, default=0.8)
        parser.add_argument('--no_cxn', action='store_true', help='if true, do **not** add batchNorm and ReLU between cnn and fc')
        parser.add_argument('--lambda_regularization', type=float, default=0.0, help='weight for feature regularization loss')
        parser.add_argument('--lambda_contrastive', type=float, default=0.0, help='weight for contrastive loss')
        parser.add_argument('--print_freq', type=int, default=50, help='print loss every print_freq iterations')
        parser.add_argument('--display_id', type=int, default=1, help='visdom window id, to disable visdom set id = -1.')
        parser.add_argument('--display_port', type=int, default=8097)
        parser.add_argument('--transforms', type=str, default='resize_affine_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--affineScale', nargs='+', type=float, default=[0.95, 1.05], help='scale tuple in transforms.RandomAffine')
        parser.add_argument('--affineDegrees', type=float, default=5, help='range of degrees in transforms.RandomAffine')
        parser.add_argument('--use_color_jitter', action='store_true', help='if specified, add color jitter in transforms')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--epoch_count', type=int, default=1, help='starting epoch')
        parser.add_argument('--min_kept', type=int, default=1)
        parser.add_argument('--max_kept', type=int, default=-1)
        parser.add_argument('--online_sourcefile', type=str, default='online.txt')
        parser.add_argument('--save_latest_freq', type=int, default=10, help='frequency of saving the latest results')
        parser.add_argument('--pair_selector_type', type=str, default='random', help='[random] | hard | easy')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')

        return parser

    def get_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(parser)
        self.opt = self.parser.parse_args()
        self.opt.use_gpu = len(self.opt.gpu_ids) > 0 and torch.cuda.is_available()
        self.opt.isTrain = self.opt.mode == 'train'
        # set age_bins
        self.opt.age_bins_with_inf = self.opt.age_bins + [float('inf')]
        # weight
        if self.opt.weight:
            assert(len(self.opt.weight) == self.opt.num_classes)
        # min_kept, max_kept
        if self.opt.max_kept < 0:
            self.opt.max_kept = self.opt.batch_size
        # num_epochs set to 1
        self.opt.num_epochs = 1
        # online_sourcefile
        if os.path.exists(os.path.join(self.opt.checkpoint_dir, self.opt.name, self.opt.online_sourcefile)):
            os.remove(os.path.join(self.opt.checkpoint_dir, self.opt.name, self.opt.online_sourcefile))
        if self.opt.mode == 'train':
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
    def __init__(self, rootdir, source_file, transform=None):
        self.rootdir = rootdir
        self.source_file = source_file
        self.transform = transform
        with open(self.source_file, 'r') as f:
            self.source_file = f.readlines()

    def __getitem__(self, index):
        s = self.source_file[index].split()
        imgA = Image.open(os.path.join(self.rootdir, s[0])).convert('RGB')
        imgB = Image.open(os.path.join(self.rootdir, s[1])).convert('RGB')
        label = int(s[2])
        if self.transform is not None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB, torch.LongTensor(1).fill_(label).squeeze(), self.source_file[index]

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
# Loss Functions
###############################################################################
# PairSelector borrowed from https://github.com/adambielski/siamese-triplet
# remark: in Adam's implementation, a batch of embeddings is first computed
# and then all possible combination of pairs are taken into consideration,
# while in this implementation, pairs are pre-generated/provided from outside,
# which gives more control over the pair-generation process.

class PairSelector:
    """
    returns indices (along Batch dimension) of pairs
    """

    def __init__(self):
        pass
    
    def get_pairs(self, scores, embeddingA, embeddingB):
        raise NotImplementedError


class RandomPairSelector(PairSelector):
    """
    Randomly selects pairs that are not equal in age
    """

    def __init__(self, min_kept=0, max_kept=0):
        super(RandomPairSelector, self).__init__()
        self._min_kept = min_kept
        self._max_kept = max_kept

    def get_pairs(self, scores, embeddingA=None, embeddingB=None):
        N = scores.size(0) * scores.size(2) * scores.size(3)
        scores = scores.detach().cpu().data.numpy()
        pred = scores.transpose((0, 2, 3, 1)).reshape(N, -1).argmax(axis=1)
        valid_inds = np.where(pred != 1)[0]
        if len(valid_inds) < self._min_kept:
            invalid_inds = np.where(pred == 1)[0]
            np.random.shuffle(invalid_inds)
            valid_inds = np.concatenate((valid_inds, invalid_inds[:self._min_kept-len(valid_inds)]))
        np.random.shuffle(valid_inds)
        selected_inds = valid_inds[:self._max_kept]
        selected_inds = torch.LongTensor(selected_inds)
        return selected_inds


class EmbeddingDistancePairSelector(PairSelector):
    """
    Selects easy pairs (pairs with largest distances)
    """

    def __init__(self, min_kept=0, max_kept=0, hard=True):
        super(EmbeddingDistancePairSelector, self).__init__()
        self._min_kept = min_kept
        self._max_kept = max_kept
        self._hard = hard  # if hard, pairs with smallest distances will be selected;
                           # if not, pairs with largest distances will be selected.

    def get_pairs(self, scores, embeddingA=None, embeddingB=None):
        N = scores.size(0) * scores.size(2) * scores.size(3)
        scores = scores.detach().cpu().data.numpy()
        pred = scores.transpose((0, 2, 3, 1)).reshape(N, -1).argmax(axis=1)
        valid_inds = np.where(pred != 1)[0]
        invalid_inds = np.where(pred == 1)[0]

        # compute square of distances
        dist2 = (embeddingA.detach()-embeddingB.detach()).pow(2)
        dist2 = dist2.cpu().data.numpy()
        dist2 = dist2.transpose((0, 2, 3, 1)).reshape(N)
        dist2_valid = dist2[valid_inds]
        dist2_invalid = dist2[invalid_inds]

        # print(dist2)
        # print(pred)

        valid_inds_sorted = valid_inds[dist2_valid.argsort()]
        invalid_inds_sorted = invalid_inds[dist2_invalid.argsort()]
        if not self._hard:
            valid_inds_sorted = valid_inds_sorted[::-1]
            invalid_inds_sorted = invalid_inds_sorted[::-1]

        all_inds = np.concatenate((valid_inds_sorted, invalid_inds_sorted))
        num_selected = min(max(len(valid_inds), self._min_kept), self._max_kept)
        selected_inds = all_inds[:num_selected]
        selected_inds = torch.LongTensor(selected_inds)
        return selected_inds


class SoftmaxPairSelector(PairSelector):
    """
    Selects hard pairs (pairs with lowest probability)
    """

    def __init__(self, min_kept=0, max_kept=0, hard=True):
        super(SoftmaxPairSelector, self).__init__()
        self._min_kept = min_kept
        self._max_kept = max_kept
        self._hard = hard  # if hard, pairs with lowest probability will be selected;
                           # if not, pairs with highest probability will be selected.

    def get_pairs(self, scores, embeddingA=None, embeddingB=None):
        N = scores.size(0) * scores.size(2) * scores.size(3)
        scores = scores.detach().cpu().permute((0, 2, 3, 1)).view(N, -1)
        probs = torch.nn.functional.softmax(scores).data.numpy()
        pred = scores.data.numpy().argmax(axis=1)
        prob = probs[range(N), pred]
        
        valid_inds = np.where(pred != 1)[0]
        invalid_inds = np.where(pred == 1)[0]

        prob_valid = prob[valid_inds]
        prob_invalid = prob[invalid_inds]

        valid_inds_sorted = valid_inds[prob_valid.argsort()]
        invalid_inds_sorted = invalid_inds[prob_invalid.argsort()]
        if not self._hard:
            valid_inds_sorted = valid_inds_sorted[::-1]
            invalid_inds_sorted = invalid_inds_sorted[::-1]

        all_inds = np.concatenate((valid_inds_sorted, invalid_inds_sorted))
        num_selected = min(max(len(valid_inds), self._min_kept), self._max_kept)
        selected_inds = all_inds[:num_selected]
        selected_inds = torch.LongTensor(selected_inds)
        return selected_inds


class OnlineCrossEntropyLoss(nn.Module):
    def __init__(self, pair_selector=None):
        super(OnlineCrossEntropyLoss, self).__init__()
        self.pair_selector = pair_selector
        self.criterion = nn.CrossEntropyLoss(weight=None)

    def __call__(self, input, target, embeddingA, embeddingB):
        N = input.size(0) * input.size(2) * input.size(3)
        target = target.reshape(input.size(0), 1, 1).expand(input.size(0), input.size(2), input.size(3))
        selected_pairs = self.pair_selector.get_pairs(input, embeddingA, embeddingB)
        if input.is_cuda:
            selected_pairs = selected_pairs.cuda()
        input = input.permute(0, 2, 3, 1).view(N, -1)
        target = target.view(N)
        loss = self.criterion(input[selected_pairs], target[selected_pairs])
        return loss, selected_pairs.cpu().data.numpy()


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight)
    
    def __call__(self, input, target):
        target_tensor = target.reshape(input.size(0), 1, 1).expand(input.size(0), input.size(2), input.size(3))
        return self.loss(input, target_tensor)


class ContrastiveLoss(nn.Module):
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
# Networks and Models
###############################################################################
class SiameseNetwork(nn.Module):
    def __init__(self, base=None, num_classes=3, pooling='avg', cnn_dim=[], cnn_pad=1, cnn_relu_slope=0.5, fc_dim=[], fc_relu_slope=0.2, dropout=0.5, no_cxn=False):
        super(SiameseNetwork, self).__init__()
        assert(num_classes == fc_dim[-1])
        self.pooling = pooling
        # base
        self.base = base
        # additional cnns
        if cnn_dim:
            conv_block = []
            nf_prev = base.feature_dim
            for i in range(len(cnn_dim)-1):
                nf = cnn_dim[i]
                conv_block += [
                    nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=cnn_pad, bias=True),
                    nn.BatchNorm2d(nf),
                    nn.LeakyReLU(cnn_relu_slope)
                ]
                nf_prev = nf
            conv_block += [nn.Conv2d(nf_prev, cnn_dim[-1], kernel_size=3, stride=1, padding=cnn_pad, bias=True)]
            self.cnn = nn.Sequential(*conv_block)
            feature_dim = cnn_dim[-1]
        else:
            self.cnn = None
            feature_dim = base.feature_dim
        # connection
        cxn_blocks = [
            nn.BatchNorm2d(cnn_dim[-1]),
            nn.LeakyReLU(cnn_relu_slope)
        ]
        self.cxn = None if no_cxn else nn.Sequential(*cxn_blocks)
        # fc layers
        fc_blocks = []
        nf_prev = feature_dim * 2
        for i in range(len(fc_dim)-1):
            nf = fc_dim[i]
            fc_blocks += [
                nn.Dropout(dropout),
                nn.Conv2d(nf_prev, nf, kernel_size=1, stride=1, padding=0, bias=True),
                nn.LeakyReLU(fc_relu_slope)
            ]
            nf_prev = nf
        fc_blocks += [
            nn.Dropout(dropout),
            nn.Conv2d(nf_prev, fc_dim[-1], kernel_size=1, stride=1, padding=0, bias=True)
        ]
        self.fc = nn.Sequential(*fc_blocks)
        self.feature_dim = feature_dim

    def forward_once(self, x):
        # FIXME: in which order? cnn -> cxn -> pooling -> fc
        output = self.base.forward(x)
        if self.cnn:
            output = self.cnn(output)
        if self.cxn:
            output = self.cxn(output)
        if self.pooling == 'avg':
            output = nn.AvgPool2d(output.size(2))(output)
        elif self.pooling == 'max':
            output = nn.MaxPool2d(output.size(2))(output)
        return output

    def forward(self, input1, input2):
        feature1 = self.forward_once(input1)
        feature2 = self.forward_once(input2)
        output = torch.cat((feature1, feature2), dim=1)
        output = self.fc(output)
        return feature1, feature2, output
    
    def load_pretrained(self, state_dict):
        # used when loading pretrained base model
        # warning: self.cnn and self.fc won't be initialized
        self.base.load_pretrained(state_dict)


class SiameseFeature(nn.Module):
    def __init__(self, base=None, pooling='avg', cnn_dim=[], cnn_pad=1, cnn_relu_slope=0.2):
        super(SiameseFeature, self).__init__()
        self.pooling = pooling
        self.base = base
        if cnn_dim:
            conv_block = []
            nf_prev = base.feature_dim
            for i in range(len(cnn_dim)-1):
                nf = cnn_dim[i]
                conv_block += [
                    nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=cnn_pad, bias=True),
                    nn.BatchNorm2d(nf),
                    nn.LeakyReLU(cnn_relu_slope)
                ]
                nf_prev = nf
            conv_block += [nn.Conv2d(nf_prev, cnn_dim[-1], kernel_size=3, stride=1, padding=cnn_pad, bias=True)]
            self.cnn = nn.Sequential(*conv_block)
            feature_dim = cnn_dim[-1]
        else:
            self.cnn = None
            feature_dim = base.feature_dim
        self.feature_dim = feature_dim
    
    def forward(self, x):
        output = self.base.forward(x)
        if self.cnn:
            output = self.cnn(output)
        if self.pooling == 'avg':
            output = nn.AvgPool2d(output.size(2))(output)
        elif self.pooling == 'max':
            output = nn.MaxPool2d(output.size(2))(output)
        return output
    
    def load_pretrained(self, state_dict):
        # load state dict from a SiameseNetwork
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        # remove cxn and fc
        for key in list(state_dict.keys()):
            if key.startswith('cxn') or key.startswith('fc'):
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=True)


class AlexNetFeature(nn.Module):
    def __init__(self, pooling='max'):
        super(AlexNetFeature, self).__init__()
        self.pooling = pooling
        sequence = [
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
        ]
        self.features = nn.Sequential(*sequence)
        self.feature_dim = 256

    def forward(self, x):
        x = self.features(x)
        if self.pooling == 'avg':
            x = nn.AvgPool2d(x.size(2))(x)
        elif self.pooling == 'max':
            x = nn.MaxPool2d(x.size(2))(x)
        return x
    
    def load_pretrained(self, state_dict):
        # invoked when used as `base' in SiameseNetwork
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        for key in list(state_dict.keys()):
            if key.startswith('classifier'):
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)


class ResNetFeature(nn.Module):
    def __init__(self, which_model):
        super(ResNetFeature, self).__init__()
        model = None
        feature_dim = None
        if which_model == 'resnet18':
            from torchvision.models import resnet18
            model = resnet18(False)
            feature_dim = 512 * 1
        elif which_model == 'resnet34':
            from torchvision.models import resnet34
            model = resnet34(False)
            feature_dim = 512 * 1
        elif which_model == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(False)
            feature_dim = 512 * 4
        elif which_model == 'resnet101':
            from torchvision.models import resnet101
            model = resnet101(False)
            feature_dim = 512 * 4
        elif which_model == 'resnet152':
            from torchvision.models import resnet152
            model = resnet152(False)
            feature_dim = 512 * 4
        delattr(model, 'fc')
        self.model = model
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x
    
    def load_pretrained(self, state_dict):
        # invoked when used as `base' in SiameseNetwork
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        self.model.load_state_dict(state_dict, strict=False)


class TheSiameseNetwork(nn.Module):
    def __init__(self):
        super(TheSiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
                
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),
            
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            
            nn.Linear(500, 5)
        )

        self.feature_dim = 5

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2, None


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
    # the original Siamese network
    if opt.which_model == 'thesiamese':
        return TheSiameseNetwork().apply(weights_init).cuda()
    
    # define base model
    base = None
    if opt.which_model == 'alexnet':
        base = AlexNetFeature(pooling='')
    elif 'resnet' in opt.which_model:
        base = ResNetFeature(opt.which_model)
    else:
        raise NotImplementedError('Model [%s] is not implemented.' % opt.which_model)
    
    # define Siamese Network
    # FIXME: SiameseNetwork or SiameseFeature according to opt.mode
    if opt.mode == 'visualize' or opt.mode == 'extract_feature':
        net = SiameseFeature(base, pooling=opt.pooling,
                             cnn_dim=opt.cnn_dim, cnn_pad=opt.cnn_pad, cnn_relu_slope=opt.cnn_relu_slope)
    else:
        net = SiameseNetwork(base, opt.num_classes, pooling=opt.pooling,
                             cnn_dim=opt.cnn_dim, cnn_pad=opt.cnn_pad, cnn_relu_slope=opt.cnn_relu_slope,
                             fc_dim=opt.fc_dim, fc_relu_slope=opt.fc_relu_slope, dropout=opt.dropout, no_cxn=opt.no_cxn)

    # initialize | load weights
    if opt.mode == 'train' and not opt.continue_train:
        net.apply(weights_init)
        if opt.pretrained_model_path:
            if isinstance(net, torch.nn.DataParallel):
                net.module.load_pretrained(opt.pretrained_model_path)
            else:
                net.load_pretrained(opt.pretrained_model_path)
    else:
        # HACK: strict=False
        net.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, opt.name, '{}_net.pth'.format(opt.which_epoch))), strict=False)
    
    if opt.mode != 'train':
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


def get_pair_selector(opt):
    if opt.pair_selector_type == 'random':
        pair_selector = RandomPairSelector(min_kept=opt.min_kept, max_kept=opt.max_kept)
    elif opt.pair_selector_type == 'hard':
        pair_selector = EmbeddingDistancePairSelector(min_kept=opt.min_kept, max_kept=opt.max_kept, hard=True)
    elif opt.pair_selector_type == 'easy':
        pair_selector = EmbeddingDistancePairSelector(min_kept=opt.min_kept, max_kept=opt.max_kept, hard=False)
    elif opt.pair_selector_type == 'softmax_hard':
        pair_selector = SoftmaxPairSelector(min_kept=opt.min_kept, max_kept=opt.max_kept, hard=True)
    elif opt.pair_selector_type == 'softmax_easy':
        pair_selector = SoftmaxPairSelector(min_kept=opt.min_kept, max_kept=opt.max_kept, hard=False)
    else:
        raise NotImplementedError
    return pair_selector


# Routines for training
def train(opt, net, dataloader, dataloader_val=None):
    # if opt.lambda_contrastive > 0:
    #     criterion_constrastive = ContrastiveLoss()
    opt.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    
    # criterion
    if len(opt.weight):
        weight = torch.Tensor(opt.weight)
        if opt.use_gpu:
            weight = weight.cuda()
    else:
        weight = None
    pair_selector = get_pair_selector(opt)
    criterion = OnlineCrossEntropyLoss(pair_selector=pair_selector)

    # optimizer
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
    cnt_online = 0
    cnt_online_diff = 0
    num_iter_per_epoch = math.ceil(dataset_size / opt.batch_size)
    opt.display_val_acc = not not dataloader_val
    loss_legend = []
    if opt.which_model != 'thesiamese':
        loss_legend.append('classification')
    if opt.lambda_contrastive > 0:
        loss_legend.append('contrastive')
    if opt.lambda_regularization > 0:
        loss_legend.append('regularization')
    if opt.display_id >= 0:
        import visdom
        vis = visdom.Visdom(server='http://localhost', port=opt.display_port)
        # plot_data = {'X': [], 'Y': [], 'leg': ['loss']}
        plot_loss = {'X': [], 'Y': [], 'leg': loss_legend}
        plot_acc = {'X': [], 'Y': [], 'leg': ['train', 'val'] if opt.display_val_acc else ['train']}
    
    # start training
    for epoch in range(opt.epoch_count, opt.num_epochs+opt.epoch_count):
        epoch_iter = 0
        pred_train = []
        target_train = []

        for i, data in enumerate(dataloader, 0):
            img0, img1, label, lines = data
            if opt.use_gpu:
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            epoch_iter += 1
            total_iter += 1

            optimizer.zero_grad()

            # net forward
            feat1, feat2, score = net(img0, img1)

            loss = 0.0
            losses = {}
            # classification loss
            if opt.which_model != 'thesiamese':
                this_loss, labeled_pairs = criterion(score, label, feat1, feat2)
                loss += this_loss
                losses['classification'] = this_loss.item()
                cnt_online += len(labeled_pairs)
                with open(os.path.join(opt.save_dir, opt.online_sourcefile), 'a') as f:
                    for j in labeled_pairs:
                        if int(lines[j].rstrip('\n').split()[2]) != 1:
                            f.write(lines[j])
                            cnt_online_diff += 1

            # regularization
            if opt.lambda_regularization > 0:
                reg1 = feat1.pow(2).mean()
                reg2 = feat2.pow(2).mean()
                this_loss = (reg1 + reg2) * opt.lambda_regularization
                loss += this_loss
                losses['regularization'] = this_loss.item()

            # get predictions
            pred_train.append(get_prediction(score))
            target_train.append(label.cpu().numpy())

            loss.backward()
            optimizer.step()

            if total_iter % opt.print_freq == 0:
                print("epoch %02d, iter %06d, loss: %.4f" % (epoch, total_iter, loss.item()))
                if opt.display_id >= 0:
                    plot_loss['X'].append(epoch+epoch_iter/num_iter_per_epoch)
                    plot_loss['Y'].append([losses[k] for k in plot_loss['leg']])
                    vis.line(
                        X=np.stack([np.array(plot_loss['X'])] * len(plot_loss['leg']), 1),
                        Y=np.array(plot_loss['Y']),
                        opts={'title': 'loss', 'legend': plot_loss['leg'], 'xlabel': 'epoch', 'ylabel': 'loss'},
                        win=opt.display_id
                    )
                loss_history.append(loss.item())
            
            if total_iter % opt.save_latest_freq == 0:
                torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, 'latest_net.pth'))
                if opt.use_gpu:
                    net.cuda()
                if epoch % opt.save_epoch_freq == 0:
                    torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, '{}_net.pth'.format(epoch)))
                    if opt.use_gpu:
                        net.cuda()
        
        curr_acc = {}
        # evaluate training
        err_train = np.count_nonzero(np.concatenate(pred_train) - np.concatenate(target_train)) / dataset_size
        curr_acc['train'] = 1 - err_train

        # evaluate val
        if opt.display_val_acc:
            pred_val = []
            target_val = []
            for i, data in enumerate(dataloader_val, 0):
                img0, img1, label, _ = data
                if opt.use_gpu:
                    img0, img1 = img0.cuda(), img1.cuda()
                _, _, output = net.forward(img0, img1)
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

    with open(os.path.join(opt.save_dir, 'loss.txt'), 'w') as f:
        for loss in loss_history:
            f.write(str(loss)+'\n')
    
    print('!!! #labeled pairs %d #diff labeld pairs %d, ratio %.2f%%' % (cnt_online, cnt_online_diff, cnt_online_diff/cnt_online*100))


# Routines for testing
def test(opt, net, dataloader):
    dataset_size_val = opt.dataset_size_val
    pred_val = []
    target_val = []
    for i, data in enumerate(dataloader, 0):
        img0, img1, label = data
        if opt.use_gpu:
            img0, img1 = img0.cuda(), img1.cuda()
        _, _, output = net.forward(img0, img1)

        pred_val.append(get_prediction(output).squeeze())
        target_val.append(label.cpu().numpy().squeeze())
        print('--> batch #%d' % (i+1))

    err = np.count_nonzero(np.stack(pred_val) - np.stack(target_val)) / dataset_size_val
    print('================================================================================')
    print('accuracy: %.6f' % (100. * (1-err)))


# Routines for visualization
def visualize(opt, net, dataloader):
    features = []
    labels = []
    for _, data in enumerate(dataloader, 0):
        img0, path0 = data
        if opt.use_gpu:
            img0 = img0.cuda()
        if opt.which_model == 'thesiamese':
            feature = net.forward_once(img0)
        else:
            feature = net.forward(img0)
        feature = feature.cpu().detach().numpy()
        features.append(feature.reshape([1, net.feature_dim]))
        labels.append(get_age_label(path0[0], opt.age_bins_with_inf))
        # np.save('features/%s' % (path0[0].replace('.jpg', '.npy')), feature)
        print('--> %s' % path0[0])

    X = np.concatenate(features, axis=0)
    labels = np.array(labels)
    np.save(os.path.join(opt.checkpoint_dir, opt.name, 'features.npy'), X)
    np.save(os.path.join(opt.checkpoint_dir, opt.name, 'labels.npy'), labels)


# Routines for visualization
def extract_feature(opt, net, dataloader):
    features = []
    labels = []
    for _, data in enumerate(dataloader, 0):
        img0, path0 = data
        if opt.use_gpu:
            img0 = img0.cuda()
        feature = net.forward(img0)
        feature = feature.cpu().detach().numpy()
        features.append(feature)
        labels.append(get_age_label(path0[0], opt.age_bins_with_inf))
        print('--> %s' % path0[0])

    X = np.concatenate(features, axis=0)
    labels = np.array(labels)
    
    # FIXME: hard coded
    new_features = []
    for L in range(5):
        idx = labels == L
        Y = X[idx,...]
        new_features.append(Y)
    np.save('features_Y.npy', new_features)
    # np.save('ex_features.npy', X)
    # np.save('ex_labels.npy', labels)


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
        dataset = SiameseNetworkDataset(opt.dataroot, opt.datafile, get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=not opt.serial_batches, num_workers=opt.num_workers, batch_size=opt.batch_size)
        opt.dataset_size = len(dataset)
        # val dataset
        if opt.dataroot_val:
            dataset_val = SiameseNetworkDataset(opt.dataroot_val, opt.datafile_val, get_transform(opt))
            dataloader_val = DataLoader(dataset_val, shuffle=False, num_workers=0, batch_size=1)
            opt.dataset_size_val = len(dataset_val)
        else:
            dataloader_val = None
            opt.dataset_size_val = 0
        print('dataset size = %d' % len(dataset))
        # train
        train(opt, net, dataloader, dataloader_val)
    elif opt.mode == 'test':
        # get dataloader
        dataset = SiameseNetworkDataset(opt.dataroot, opt.datafile, transform=get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=opt.num_workers, batch_size=opt.batch_size)
        opt.dataset_size_val = len(dataset)
        # test
        test(opt, net, dataloader)
    elif opt.mode == 'visualize':
        # get dataloader
        dataset = SingleImageDataset(opt.dataroot, opt.datafile, transform=get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)
        # visualize
        visualize(opt, net, dataloader)
    elif opt.mode == 'extract_feature':
        # get dataloader
        dataset = SingleImageDataset(opt.dataroot, opt.datafile, transform=get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)
        # test
        extract_feature(opt, net, dataloader)
    else:
        raise NotImplementedError('Mode [%s] is not implemented.' % opt.mode)

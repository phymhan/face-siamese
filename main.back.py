import os
import argparse
import random
import functools
import numpy as np
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
        parser.add_argument('--pretrained_model_path', type=str, default='pretrained_models/alexnet.pth', help='path to pretrained models')
        parser.add_argument('--use_pretrained_model', action='store_true', help='use pretrained model')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints')
        parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
        parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
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
        parser.add_argument('--fc_dim', type=int, default=64, help='dimension of fc')
        parser.add_argument('--cnn_dim', type=int, nargs='+', default=[], help='cnn kernel dims for feature dimension reduction')
        parser.add_argument('--lambda_regularization', type=float, default=0.0, help='weight for feature regularization loss')
        parser.add_argument('--lambda_contrastive', type=float, default=0.0)

        return parser

    def get_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(parser)
        self.opt = self.parser.parse_args()
        self.opt.use_gpu = len(self.opt.gpu_ids) > 0 and torch.cuda.is_available()
        # set age_bins
        self.opt.age_bins_with_inf = self.opt.age_bins + [float('inf')]
        # weight
        if self.opt.weight:
            assert(len(self.opt.weight) == self.opt.num_classes)
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
        return imgA, imgB, torch.LongTensor(1).fill_(label).squeeze()

    def __len__(self):
        # # shuffle source file
        # random.shuffle(self.source_file)
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
class SiameseNetwork(nn.Module):
    def __init__(self, num_classes=3, base=None, pooling='avg', dropout=0.5, fc_dim=64, cnn_dim=[]):
        super(SiameseNetwork, self).__init__()
        self.pooling = pooling
        fw = 1 if pooling else 6
        # base
        self.base = base
        # additional cnns
        if cnn_dim:
            conv_block = []
            nf_prev = base.feature_dim
            for i in range(len(cnn_dim)-1):
                nf = cnn_dim[i]
                conv_block += [
                    nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(nf),
                    nn.ReLU(True)
                ]
                nf_prev = nf
            nf = cnn_dim[-1]
            conv_block += [nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=1, bias=True)]
            base.feature_dim = nf
            self.cnn = nn.Sequential(*conv_block)
            self.cxn = nn.Sequential(
                nn.BatchNorm2d(nf),
                nn.ReLU(True)
            )
        else:
            self.cnn = None
            self.cxn = None
        # fc layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base.feature_dim * fw * fw * 2, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )
        self.feature_dim = base.feature_dim

    def forward_once(self, x):
        output = self.base.forward(x)
        if self.cnn:
            output = self.cnn(output)
        feature = output
        if self.cxn:
            output = self.cxn(output)
        if self.pooling == 'avg':
            output = nn.AvgPool2d(output.size(2))(output)
        elif self.pooling == 'max':
            output = nn.MaxPool2d(output.size(2))(output)
        output = output.view(output.size(0), -1)
        return output, nn.AvgPool2d(feature.size(2))(feature)

    def forward(self, input1, input2):
        output1, feature1 = self.forward_once(input1)
        output2, feature2 = self.forward_once(input2)
        output = torch.cat((output1, output2), dim=1)
        output = self.fc(output)
        return feature1, feature2, output
    
    def load_pretrained(self, state_dict):
        # used when loading pretrained base model
        # warning: self.cnn and self.fc won't be initialized
        self.base.load_pretrained(state_dict)


class SiameseFeature(nn.Module):
    def __init__(self, num_classes=3, base=None, pooling='avg', dropout=0.5, fc_dim=64, cnn_dim=[]):
        super(SiameseFeature, self).__init__()
        self.pooling = pooling
        self.base = base
        if cnn_dim:
            conv_block = []
            nf_prev = base.feature_dim
            for i in range(len(cnn_dim)-1):
                nf = cnn_dim[i]
                conv_block += [
                    nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(nf),
                    nn.ReLU(True)
                ]
                nf_prev = nf
            nf = cnn_dim[-1]
            conv_block += [nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=1, bias=True)]
            base.feature_dim = nf
            self.cnn = nn.Sequential(*conv_block)
        else:
            self.cnn = None
        self.feature_dim = base.feature_dim
    
    def forward(self, x):
        output = self.base.forward(x)
        if self.cnn:
            output = self.cnn(output)
        if self.pooling == 'avg':
            output = nn.AvgPool2d(output.size(2))(output)
        elif self.pooling == 'max':
            output = nn.MaxPool2d(output.size(2))(output)
        return output


class AlexNetFeature(nn.Module):
    def __init__(self, pooling='avg'):
        super(AlexNetFeature, self).__init__()
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
        if pooling == 'avg':
            sequence += [nn.AvgPool2d(kernel_size=6)]
        elif pooling == 'max':
            sequence += [nn.MaxPool2d(kernel_size=6)]
        self.features = nn.Sequential(*sequence)
        self.feature_dim = 256

    def forward(self, x):
        x = self.features(x)
        return x
    
    def load_pretrained(self, state_dict):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
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


def parse_age_label(fname, binranges):
    strlist = fname.split('_')
    age = int(strlist[0])
    l = None
    for l in range(len(binranges)-1):
        if (age >= binranges[l]) and (age < binranges[l+1]):
            break
    return l


###############################################################################
# Main Routines
###############################################################################
def get_model(opt):
    # the original Siamese network
    if opt.which_model == 'thesiamese':
        net = TheSiameseNetwork()
        net.apply(weights_init)
        return net.cuda()
    
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
        net = SiameseFeature(opt.num_classes, base, opt.pooling, opt.dropout, opt.fc_dim, opt.cnn_dim)
    else:
        net = SiameseNetwork(opt.num_classes, base, opt.pooling, opt.dropout, opt.fc_dim, opt.cnn_dim)

    # initialize | load weights
    if opt.mode == 'train':
        net.apply(weights_init)
        if opt.use_pretrained_model:
            if isinstance(net, torch.nn.DataParallel):
                net.module.load_pretrained(opt.pretrained_model_path)
            else:
                net.load_pretrained(opt.pretrained_model_path)
    else:
        # HACK: strict=False
        net.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, opt.name, '{}_net.pth'.format(opt.which_epoch))), strict=False)
        net.eval()
    
    if opt.use_gpu:
        net.cuda()
    return net


# Routines for training
def train(opt, net, dataloader):
    if opt.use_contrastive_loss:
        criterion_constrastive = ContrastiveLoss()
    opt.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    # criterion
    if len(opt.weight):
        opt.weight = torch.Tensor(opt.weight)
        if opt.use_gpu:
            opt.weight = opt.weight.cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=opt.weight)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # optimizer
    if opt.finetune_fc_only:
        if isinstance(net, nn.DataParallel):
            param = net.module.get_finetune_parameters()
        else:
            param = net.get_finetune_parameters()
    else:
        param = net.parameters()
    optimizer = optim.Adam(param, lr=opt.lr)

    counter = []
    loss_history = []
    iteration_number = 0

    # start training
    for epoch in range(1, opt.num_epochs+1):
        for i, data in enumerate(dataloader, 0):
            img0, img1, label = data
            if opt.use_gpu:
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()
            # classification loss
            if opt.which_model != 'thesiamese':
                feat1, feat2, pred = net(img0, img1)
                loss_classification = criterion(pred, label)
            else:
                loss_classification = 0

            # contrastive loss
            if opt.lambda_contrastive > 0:
                # new label: 0 similar (1), 1 dissimilar (0, 2)
                idx = label == 1
                label_new = label.clone()
                label_new[idx] = 0
                label_new[1-idx] = 1
                loss_contrastive = criterion_constrastive(
                    feat1.view(feat1.size(0), -1), feat2.view(feat1.size(0), -1), label_new.float()
                    ) * opt.lambda_contrastive
            else:
                loss_contrastive = 0
            
            # regularization
            if opt.lambda_regularization > 0:
                reg1 = feat1.pow(2).mean()
                reg2 = feat2.pow(2).mean()
                loss_regularization = (reg1 + reg2) * opt.lambda_regularization
            else:
                loss_regularization = 0
            
            # conbined loss
            loss = loss_classification + loss_contrastive + loss_regularization

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.item())
        if epoch % opt.save_epoch_freq == 0:
            torch.save(net.state_dict(), os.path.join(opt.save_dir, '{}_net.pth'.format(epoch)))

    # save model
    torch.save(net.state_dict(), os.path.join(opt.save_dir, 'latest_net.pth'))

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
        img0, img1, label = data
        if opt.use_gpu:
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
        output, _, _ = net.forward(img0, img1)
        target.append(label.cpu().detach().numpy().squeeze())
        pred.append(output.cpu().detach().numpy().squeeze().argmax())
        if target[-1] == pred[-1]:
            acc += 1
        print('--> image #%d: target %d   pred %d' % (i+1, target[-1], pred[-1]))

    print('================================================================================')
    print('accuracy: %.6f' % (100. * acc/len(pred)))


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
        labels.append(parse_age_label(path0[0], opt.age_bins_with_inf))
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
        labels.append(parse_age_label(path0[0], opt.age_bins_with_inf))
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
        dataset = SiameseNetworkDataset(opt.dataroot, opt.datafile, transform=transforms.Compose([transforms.Resize((opt.loadSize, opt.loadSize)), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        dataloader = DataLoader(dataset, shuffle=True, num_workers=opt.num_workers, batch_size=opt.batch_size)
        # train
        train(opt, net, dataloader)
    elif opt.mode == 'test':
        # get dataloader
        dataset = SiameseNetworkDataset(opt.dataroot, opt.datafile, transform=transforms.Compose([transforms.Resize((opt.loadSize, opt.loadSize)), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=1, batch_size=1)
        # test
        test(opt, net, dataloader)
    elif opt.mode == 'visualize':
        # get dataloader
        dataset = SingleImageDataset(opt.dataroot, opt.datafile, transform=transforms.Compose([transforms.Resize((opt.loadSize, opt.loadSize)), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=1, batch_size=1)
        # test
        visualize(opt, net, dataloader)
    elif opt.mode == 'extract_feature':
        # get dataloader
        dataset = SingleImageDataset(opt.dataroot, opt.datafile, transform=transforms.Compose([transforms.Resize((opt.loadSize, opt.loadSize)), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=1, batch_size=1)
        # test
        extract_feature(opt, net, dataloader)
    else:
        raise NotImplementedError('Mode [%s] is not implemented.' % opt.mode)

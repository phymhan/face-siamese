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
        parser.add_argument('--fineSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--age_bins', nargs='+', type=int, default=[1, 11, 21, 31, 41, 51, 61, 71, 81, 91], help='list of bins, the (i+1)-th group is in the range [age_binranges[i], age_binranges[i+1]), e.g. [1, 11, 21, ..., 101], the 1-st group is [1, 10], the 9-th [91, 100], however, the 10-th [101, +inf)')
        parser.add_argument('--print_freq', type=int, default=50, help='print loss every print_freq iterations')
        parser.add_argument('--display_id', type=int, default=1, help='visdom window id, to disable visdom set id = -1.')
        parser.add_argument('--display_port', type=int, default=8097)
        parser.add_argument('--delta', type=float, default=0.05)
        parser.add_argument('--embedding_mean', type=float, default=0)
        parser.add_argument('--embedding_std', type=float, default=1)
        parser.add_argument('--cnn_dim', type=int, nargs='+', default=[64, 1], help='cnn kernel dims for feature dimension reduction')
        parser.add_argument('--cnn_pad', type=int, default=1, help='padding of cnn layers defined by cnn_dim')
        parser.add_argument('--cnn_relu_slope', type=float, default=0.8)
        parser.add_argument('--transforms', type=str, default='resize_affine_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--affineScale', nargs='+', type=float, default=[0.95, 1.05], help='scale tuple in transforms.RandomAffine')
        parser.add_argument('--affineDegrees', type=float, default=5, help='range of degrees in transforms.RandomAffine')
        parser.add_argument('--use_color_jitter', action='store_true', help='if specified, add color jitter in transforms')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--decoder_dim', type=int, nargs='+', default=[4, 4, 1], help='cnn kernel dims for feature dimension reduction')
        parser.add_argument('--decoder_relu_slope', type=float, default=0.2)
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--siamese_model_path', type=str, default='pretrained_models/siamese.pth')
        

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
        # embedding normalization
        self.opt.embedding_normalize = lambda x: (x - self.opt.embedding_mean) / self.opt.embedding_std
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
class EmbeddingDecoder(nn.Module):
    def __init__(self, embedding_dim=1, cnn_dim=[], cnn_relu_slope=0.2):
        super(EmbeddingDecoder, self).__init__()
        conv_block = []
        nf_prev = embedding_dim
        for i in range(len(cnn_dim)-1):
            nf = cnn_dim[i]
            conv_block += [
                nn.Conv2d(nf_prev, nf, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(cnn_relu_slope)
            ]
            nf_prev = nf
        conv_block += [nn.Conv2d(nf_prev, cnn_dim[-1], kernel_size=1, stride=1, padding=0, bias=True)]
        self.cnn = nn.Sequential(*conv_block)
    
    def forward(self, x):
        return self.cnn(x)


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
    age = float(strlist[0])
    l = None
    for l in range(len(binranges)-1):
        if (age >= binranges[l]) and (age < binranges[l+1]):
            break
    return l


def get_age(fname):
    strlist = fname.split('_')
    age = float(strlist[0])
    return age


def get_prediction(score):
    batch_size = score.size(0)
    score_cpu = score.detach().cpu().numpy()
    pred = stats.mode(score_cpu.argmax(axis=1).reshape(batch_size, -1), axis=1)
    return pred[0].reshape(batch_size)


def get_accuracy(pred, target, delta):
    acc = torch.abs(pred.cpu() - target.cpu()) < delta
    return acc.view(-1).numpy()


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


###############################################################################
# Main Routines
###############################################################################
def get_decoder(opt):
    net = EmbeddingDecoder(opt.embedding_dim, opt.decoder_dim, opt.decoder_relu_slope)
    if opt.mode == 'train' and not opt.continue_train:
        net.apply(weights_init)
    else:
        net.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, opt.name, '{}_decoder.pth'.format(opt.which_epoch))), strict=False)
    
    if opt.mode != 'train':
        net.eval()
    
    if opt.use_gpu:
        net.cuda()
    return net


def get_model(opt):
    # define base model
    base = None
    if opt.which_model == 'alexnet':
        base = AlexNetFeature(pooling='')
    elif 'resnet' in opt.which_model:
        base = ResNetFeature(opt.which_model)
    else:
        raise NotImplementedError('Model [%s] is not implemented.' % opt.which_model)
    
    # define SiameseFeature
    net = SiameseFeature(base, pooling=opt.pooling,
                         cnn_dim=opt.cnn_dim, cnn_pad=opt.cnn_pad, cnn_relu_slope=opt.cnn_relu_slope)
    
    net.load_state_dict(torch.load(opt.siamese_model_path), strict=False)
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
def train(opt, net, decoder, dataloader):
    criterion = torch.nn.MSELoss()
    opt.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    optimizer = optim.Adam(decoder.parameters(), lr=opt.lr)

    dataset_size, dataset_size_val = opt.dataset_size, opt.dataset_size_val
    loss_history = []
    total_iter = 0
    num_iter_per_epoch = math.ceil(dataset_size / opt.batch_size)
    opt.display_val_acc = not not dataloader_val
    loss_legend = ['regression']
    if opt.display_id >= 0:
        import visdom
        vis = visdom.Visdom(server='http://localhost', port=opt.display_port)
        plot_loss = {'X': [], 'Y': [], 'leg': loss_legend}
        plot_acc = {'X': [], 'Y': [], 'leg': ['train', 'val'] if opt.display_val_acc else ['train']}

    for epoch in range(opt.epoch_count, opt.num_epochs+opt.epoch_count):
        epoch_iter = 0
        acc_train = []

        for i, data in enumerate(dataloader, 0):
            img0, path0 = data
            label = [get_age(name) for name in path0]
            label = torch.FloatTensor(label).view(img0.size(0), net.feature_dim, 1, 1)

            # normalize label only
            label = opt.embedding_normalize(label)

            if opt.use_gpu:
                img0, label = img0.cuda(), label.cuda()
            epoch_iter += 1
            total_iter += 1

            optimizer.zero_grad()

            embedding = net.forward(img0)
            output = decoder(embedding)
            loss = criterion(output, label)

            # get predictions
            acc_train.append(get_accuracy(output, label, opt.delta))

            loss.backward()
            optimizer.step()

            losses = {'regression': loss.item()}
            
            if total_iter % opt.print_freq == 0:
                print("epoch %02d, iter %06d, loss: %.4f" % (epoch, total_iter, loss.item()))
                if opt.display_id >= 0:
                    plot_loss['X'].append(epoch -1 + epoch_iter/num_iter_per_epoch)
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
        curr_acc['train'] = np.count_nonzero(np.concatenate(acc_train)) / dataset_size

        # evaluate val
        if opt.display_val_acc:
            acc_val = []
            for i, data in enumerate(dataloader_val, 0):
                img0, path0 = data
                label = [get_age(name) for name in path0]
                label = torch.FloatTensor(label).view(img0.size(0), net.feature_dim, 1, 1)

                # normalize label only
                label = opt.embedding_normalize(label)

                if opt.use_gpu:
                    img0 = img0.cuda()
                embedding = net.forward(img0)
                output = decoder.forward(embedding)
                acc_val.append(get_accuracy(output, label, opt.delta))
            curr_acc['val'] = np.count_nonzero(np.concatenate(acc_val)) / dataset_size_val
        
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

        torch.save(decoder.cpu().state_dict(), os.path.join(opt.save_dir, 'latest_decoder.pth'))
        if opt.use_gpu:
            decoder.cuda()
        if epoch % opt.save_epoch_freq == 0:
            torch.save(decoder.cpu().state_dict(), os.path.join(opt.save_dir, '{}_decoder.pth'.format(epoch)))
            if opt.use_gpu:
                decoder.cuda()

    with open(os.path.join(opt.save_dir, 'loss.txt'), 'w') as f:
        for loss in loss_history:
            f.write(str(loss)+'\n')
    # show_plot(counter, loss_history)


# Routines for visualization
def visualize(opt, net, decoder, dataloader):
    outputs = []
    labels = []
    for _, data in enumerate(dataloader, 0):
        img0, path0 = data
        if opt.use_gpu:
            img0 = img0.cuda()
        embedding = net.forward(img0)
        output = decoder.forward(embedding)
        output = output.cpu().detach().numpy()
        outputs.append(output.reshape([1, net.feature_dim]))
        labels.append(get_age(path0[0]))
        print('--> %s' % path0[0])

    X = np.concatenate(outputs, axis=0)
    labels = np.array(labels)
    np.save(os.path.join(opt.checkpoint_dir, opt.name, 'outputs.npy'), X)
    np.save(os.path.join(opt.checkpoint_dir, opt.name, 'labels.npy'), labels)


###############################################################################
# main()
###############################################################################
# TODO: set random seed

if __name__=='__main__':
    opt = Options().get_options()

    # get model
    net = get_model(opt)
    set_requires_grad(net, False)
    opt.embedding_dim = net.feature_dim
    decoder = get_decoder(opt)

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
        train(opt, net, decoder, dataloader)
    elif opt.mode == 'visualize':
        set_requires_grad(decoder, False)
        # get dataloader
        dataset = SingleImageDataset(opt.dataroot, opt.datafile, transform=get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)
        # visualize
        visualize(opt, net, decoder, dataloader)
    else:
        raise NotImplementedError('Mode [%s] is not implemented.' % opt.mode)

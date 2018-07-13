import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import argparse


class Options():
    def initialize(self, parser):
        parser.add_argument('--mode', type=str, default='train', help='train | test | visualize')
        parser.add_argument('--name', type=str, default='exp', help='experiment name')
        parser.add_argument('--dataroot', required=True, help='path to images')
        parser.add_argument('--datafile', type=str, default='', help='text file listing images')
        parser.add_argument('--pretrained_model_dir', type=str, default='pretrained_models', help='pretrained models')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints')
        parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
        parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        return parser

    def get_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(parser)
        self.opt = self.parser.parse_args()
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
        with open(source_file, 'r') as f:
            self.source_file = f.readlines()
        # if os.path.isdir(rootdir):
        #     self.source_file = os.listdir(rootdir)
        # else:
        #     with open(rootdir, 'r') as f:
        #         self.source_file = f.readlines()

    def __getitem__(self, index):
        imgA = Image.open(os.path.join(self.rootdir, self.source_file[index].rstrip('\n'))).convert('RGB')

        if self.transform is not None:
            imgA = self.transform(imgA)
        return imgA, self.source_file[index]

    def __len__(self):
        return len(self.source_file)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = AlexNet(num_classes=3)
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 1 * 1 * 2, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 3),
        )

    def forward_once(self, x):
        output = self.cnn1.forward(x)
        output = nn.AvgPool2d(6)(output)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.cat([output1, output2], dim=1)
        output = self.fc1(output)
        return output


class AlexNet(nn.Module):
    def __init__(self, num_classes=3):
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
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 256 * 6 * 6)
        # x = self.classifier(x)
        return x


class AlexNet2(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet2, self).__init__()
        self.cnn1 = AlexNet()
        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(500, num_classes),
        )


    def forward(self, x):
        x = self.cnn1.forward(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc1(x)
        return x


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
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def parse_label(names, num_classes=10):
    labels = []
    for name in names:
        s = name.split('_')
        l = (int(s[0])-1) // 10
        l = max(0, min(num_classes-1, l))
        labels.append(l)
    return labels


def get_dataloader(opt):
    dataset = SingleImageDataset(opt.dataroot, opt.datafile, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    dataloader = DataLoader(dataset, shuffle=True, num_workers=opt.num_workers, batch_size=opt.batch_size)
    return dataloader


def get_model(opt):
    net = AlexNet2(10)
    net.cnn1.load_state_dict(torch.load(os.path.join(opt.pretrained_model_dir, 'alexnet.pth')), strict=False)
    net.fc1.apply(weights_init)
    net.cuda()
    return net


def train(opt, net, dataloader):
    opt.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, opt.num_epochs):
        for i, data in enumerate(dataloader, 0):
            img0, path0 = data
            label = parse_label(path0)
            label = torch.LongTensor(label)
            img0, label = img0.cuda(), label.cuda()
            optimizer.zero_grad()
            output = net.forward(img0)
            loss = criterion(output, label)
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

    with open(os.path.join(opt.save_dir, 'log.txt'), 'w') as f:
        for loss in loss_history:
            f.write(str(loss)+'\n')
    # show_plot(counter, loss_history)


if __name__=='__main__':
    opt = Options().get_options()

    # Get dataloader
    dataloader = get_dataloader(opt)

    # Get model
    model = get_model(opt)

    if opt.mode == 'train':
        train(opt, model, dataloader)

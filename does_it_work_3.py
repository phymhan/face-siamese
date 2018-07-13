import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
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


class Config():
    training_dir = "/media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400/train"
    testing_dir = "/media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400/test"
    training_source_file = "/media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400/train_pairs_same.txt"
    alexnet_pretrained_model_path = "/media/ligong/Toshiba/Research/buddha-sketch/pretrained_models/alexnet-owt-4df8aa71.pth"
    checkpoint_dir = "./checkpoints3"
    save_epoch_freq = 10
    train_batch_size = 64
    train_number_epochs = 100
    num_workers = 8
    init_type = 'normal'
    gpu_ids = [0]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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

    def __init__(self, rootdir, transform=None):
        self.rootdir = rootdir
        self.transform = transform
        self.source_file = os.listdir(rootdir)

    def __getitem__(self, index):
        imgA = Image.open(os.path.join(self.rootdir, self.source_file[index])).convert('RGB')

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


def parse_label(names):
    labels = []
    for name in names:
        ss = name.split('_')
        l = (int(ss[0])-1) // 10 - 1
        l = max(0, min(4, l))
        labels.append(l)
    return labels


siamese_dataset = SingleImageDataset(Config.testing_dir,
                                     transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                   transforms.ToTensor()]))

test_dataloader = DataLoader(siamese_dataset,
                              shuffle=False,
                              num_workers=1,
                              batch_size=1)
net = AlexNet2(5)
net.load_state_dict(torch.load('./checkpoints4/checkpoint_30.pth.tar')['state_dict'])
net.cuda()

pred = []
target = []
acc = 0

for i, data in enumerate(test_dataloader, 0):
    img0, path0 = data
    label = parse_label(path0)
    label = torch.LongTensor(label)
    img0, label = img0.cuda(), label.cuda()
    output = net.forward(img0)
    target.append(label.cpu().detach().numpy().squeeze())
    pred.append(output.cpu().detach().numpy().squeeze().argmax())
    if target[-1] == pred[-1]:
        acc += 1
    # loss = criterion(output, label.squeeze(1))
    print('--> pair #%d: target %d   pred %d' % (i+1, target[-1], pred[-1]))

print('=' * 100)
print('accuracy: %f' % (100. * acc/len(pred)))


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
    training_source_file = "/media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400/train_pairs_diff.txt"
    alexnet_pretrained_model_path = "pretrained_models/alexnet-owt-4df8aa71.pth"
    checkpoint_dir = "./checkpoints2"
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
    elif classname.find('Linear') != -1:
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


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = AlexNet(num_classes=3)
        self.cnn2 = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(6),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2 * 1 * 1 * 2, 4),
            nn.ReLU(inplace=True),

            nn.Linear(4, 3),
        )

    def get_feature(self, x):
        output = self.cnn1.forward(x)
        output = self.cnn2(output)
        return output

    def forward_once(self, x):
        output = self.cnn1.forward(x)
        output = self.cnn2(output)
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


siamese_dataset = SiameseNetworkDataset(Config.training_dir, Config.training_source_file,
                                        transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                      transforms.ToTensor()]))

train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              num_workers=Config.num_workers,
                              batch_size=Config.train_batch_size)
net = SiameseNetwork()
net.cnn1.load_state_dict(torch.load(Config.alexnet_pretrained_model_path), strict=False)
net.cnn2.apply(weights_init)
net.fc1.apply(weights_init)
net.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number = 0

if not os.path.exists(Config.checkpoint_dir):
    os.makedirs(Config.checkpoint_dir)

for epoch in range(0, Config.train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
        optimizer.zero_grad()
        output = net.forward(img0, img1)
        loss = criterion(output, label.squeeze(1))
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss.item())
    if epoch % Config.save_epoch_freq == 0:
        torch.save({'state_dict': net.state_dict(), 'epoch': epoch}, os.path.join(Config.checkpoint_dir, 'checkpoint_{}.pth.tar'.format(epoch)))

# save model
torch.save({'state_dict': net.state_dict(), 'epoch': epoch}, os.path.join(Config.checkpoint_dir, 'checkpoint_latest.pth.tar'))

show_plot(counter, loss_history)

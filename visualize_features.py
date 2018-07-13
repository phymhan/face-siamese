import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image


class Config():
    training_dir = "/media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400/train"
    testing_dir = "/media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400/test"
    training_source_file = "/media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400/train_pairs_diff.txt"
    testing_source_file = "/media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400/test_pairs_diff.txt"
    alexnet_pretrained_model_path = "/media/ligong/Toshiba/Research/buddha-sketch/pretrained_models/alexnet-owt-4df8aa71.pth"
    checkpoint_dir = "./features"
    save_epoch_freq = 10
    train_batch_size = 64
    train_number_epochs = 100
    num_workers = 8
    init_type = 'normal'
    gpu_ids = [0]


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


# build model
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
# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()
#         self.cnn1 = AlexNet(num_classes=3)
#         self.fc1 = nn.Sequential(
#             nn.Linear(256 * 1 * 1 * 2, 500),
#             nn.ReLU(inplace=True),

#             nn.Linear(500, 500),
#             nn.ReLU(inplace=True),

#             nn.Linear(500, 3),
#         )

#     def forward_once(self, x):
#         output = self.cnn1.forward(x)
#         output = nn.AvgPool2d(6)(output)
#         output = output.view(output.size()[0], -1)
#         return output

#     def forward(self, input1, input2):
#         output1 = self.forward_once(input1)
#         output2 = self.forward_once(input2)
#         output = torch.cat([output1, output2], dim=1)
#         output = self.fc1(output)
#         return output


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


def parse_label(name):
    ss = name.split('_')
    l = (int(ss[0])-1) // 10 - 1
    l = max(0, min(4, l))
    return l


siamese_dataset = SingleImageDataset(Config.testing_dir,
                                     transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                   transforms.ToTensor()]))

test_dataloader = DataLoader(siamese_dataset,
                             shuffle=False,
                             num_workers=1,
                             batch_size=1)


# load model
net = SiameseNetwork()
net.load_state_dict(torch.load('./checkpoints2/checkpoint_latest.pth.tar')['state_dict'])
net.cuda()
pred = []
target = []
acc = 0

if not os.path.exists(Config.checkpoint_dir):
    os.makedirs(Config.checkpoint_dir)

features = []
labels = []
for i, data in enumerate(test_dataloader, 0):
    img0, path0 = data
    feature = net.get_feature(img0.cuda())
    feature = feature.cpu().detach().numpy()
    features.append(feature.reshape([1, 2]))
    labels.append(parse_label(path0[0]))
    # np.save('features/%s' % (path0[0].replace('.jpg', '.npy')), feature)
    print('--> %s' % path0[0])

X = np.concatenate(features, axis=0)
labels = np.array(labels)
np.save('features.npy', X)
np.save('labels.npy', labels)
colors = 'r', 'g', 'b', 'c', 'm'
from matplotlib import pyplot as plt
plt.figure()
for i, c, label in zip([0, 1, 2, 3, 4], colors, [0, 1, 2, 3, 4]):
    plt.scatter(X[labels==i, 0], X[labels==i, 1], c=c, label=label)

# plt.scatter(X[:,0], X[:,1], label=labels)
plt.legend()
plt.show()

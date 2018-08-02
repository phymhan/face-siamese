# datafile: A B 0/1/2
# label: 0: A < B, 1: A == B, 2: A > B

import os
import random
import argparse

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--N', type=int, default=3461)
parser.add_argument('--margin', type=int, default=10)
opt = parser.parse_args()

def parse_age_label(fname, binranges):
    strlist = fname.split('_')
    age = int(strlist[0])
    l = None
    for l in range(len(binranges)-1):
        if (age >= binranges[l]) and (age < binranges[l+1]):
            break
    return l

def parse_age(fname):
    strlist = fname.split('_')
    age = int(strlist[0])
    return age

root = '/media/ligong/Toshiba/Datasets/UTKFace'
mode = opt.mode
src = '../data/'+mode+'.txt'
N = opt.N
# binranges = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91] + [float('inf')]
binranges = [1, 21, 41, 61, 81] + [float('inf')]
num_classes = len(binranges)-1

paths = [[] for _ in range(num_classes)]
with open(src, 'r') as f:
    fnames = f.readlines()
for id, fname in enumerate(fnames):
    fname = fname.rstrip('\n').split()[0]
    label = parse_age_label(fname, binranges)
    paths[label].append(fname)

def label_fn(a1, a2, m):
    if abs(a1-a2) <= m:
        return 1
    elif a1 < a2:
        return 0
    else:
        return 2

with open(mode+'_pairs_plus.txt', 'w') as f:
    for _ in range(N):
        l1 = 4
        name1 = random.choice(paths[l1])
        name2 = random.choice(fnames).rstrip('\n')
        if random.random() < 0.5:
            tmp = name1
            name1 = name2
            name2 = tmp
        label = label_fn(parse_age(name1), parse_age(name2), opt.margin)
        f.write('%s %s %d\n' % (name1, name2, label))

print('$ python generate_pairs.py\n$ python generate_pairs3.py\n$ cat train_pairs_plus.txt >> train_pairs.txt')


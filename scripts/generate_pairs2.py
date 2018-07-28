# datafile: A B 0/1/2
# label: 0: A < B, 1: A == B, 2: A > B

import os
import random
import argparse

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--N', type=int, default=24000)
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

with open(src, 'r') as f:
    fnames = f.readlines()

def label_fn(a1, a2, m):
    if abs(a1-a2) <= m:
        return 1
    elif a1 < a2:
        return 0
    else:
        return 2

cnt = [0, 0, 0]

with open(mode+'_pairs.txt', 'w') as f:
    for _ in range(N):
        ss = random.sample(fnames, 2)
        name1 = ss[0].rstrip('\n')
        name2 = ss[1].rstrip('\n')
        label = label_fn(parse_age(name1), parse_age(name2), opt.margin)
        cnt[label] += 1
        f.write('%s %s %d\n' % (name1, name2, label))

w = []
for c in cnt:
    w.append(1.0 * sum(cnt) / c)

print([x/sum(w) for x in w])


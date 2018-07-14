import os
import random
from itertools import chain

def parse_age(fname, max_age=9, min_age=0):
    s = fname.split('_')
    l = (int(s[0])-1)//10
    l = max(min(l, max_age), min_age)
    return l

random.seed(0)
src = '/media/ligong/Toshiba/Datasets/UTKFace'

cnt = {}
name_list = {}
for l in range(10):
    cnt[l] = 0
    name_list[l] = []

for fname in os.listdir(src):
    l = parse_age(fname)
    cnt[l] += 1
    name_list[l].append(fname)

print(cnt)

test_num = {}
for l in range(10):
    if cnt[l] < 1000:
        test_num[l] = int(cnt[l] * 0.2)
    else:
        test_num[l] = 200

train_list = {}
test_list = {}
for l in range(10):
    random.shuffle(name_list[l])
    test_list[l] = name_list[l][:test_num[l]]
    train_list[l] = name_list[l][test_num[l]:]

with open('../data/train.txt', 'w') as f:
    for name in list(chain.from_iterable(train_list.values())):
        f.write(name+'\n')

with open('../data/test.txt', 'w') as f:
    for name in list(chain.from_iterable(test_list.values())):
        f.write(name+'\n')

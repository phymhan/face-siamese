import os

def parse_age(fname, max_age=9, min_age=0):
    s = fname.split('_')
    l = (int(s[0])-1)//10
    l = max(min(l, max_age), min_age)
    return l

src = '/data/home/ligonghan/Research/Datasets/UTKFace'

cnt = {}
for l in range(10):
    cnt[l] = 0

for fname in os.listdir(src):
    l = parse_age(fname)
    cnt[l] += 1

print(cnt)

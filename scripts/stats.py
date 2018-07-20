import os

def parse_age(fname, max_age=4, min_age=0):
    s = fname.split('_')
    l = (int(s[0])-1)//20
    l = max(min(l, max_age), min_age)
    return l

src = '/media/ligong/Toshiba/Datasets/UTKFace'

cnt = [0 for _ in range(5)]

for fname in os.listdir(src):
    l = parse_age(fname)
    cnt[l] += 1

w = []
for c in cnt:
    w.append(1.0 * sum(cnt) / c)

print([x/sum(w) for x in w])

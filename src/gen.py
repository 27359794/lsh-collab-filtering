from collections import defaultdict
import os

root = '..'
training_set = 'training_set'

MIDS = range(1, 3001)

f = open(os.path.join(root, 'probe.txt'))
probe = defaultdict(list)
mid = None
for line in f:
    if ':' in line:
        mid = int(line.split(':')[0])
    else:
        uid = int(line)
        probe[mid].append(uid)

w = open(os.path.join(root, 'probe_rated_{}_{}.txt'.format(training_set, len(MIDS))), 'w')

for mid in MIDS:
    name = 'mv_{:0>7}.txt'.format(mid)

    if mid in probe:
        w.write('%d:\n' % mid)

    f = open(os.path.join(root, training_set, name))
    tf = open(os.path.join(root, 'noprobe', training_set, name), 'w')
    tf.write(f.readline())
    for line in f:
        u, r, _ = line.split(',')
        u = int(u)
        r = int(r)
        if u in probe[mid]:
            w.write(line)
        else:
            tf.write(line)


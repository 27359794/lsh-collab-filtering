from collections import defaultdict

root = '../'
training_set = 'training_med'

MIDS = range(1, 17771)

f = open(root + 'probe.txt')
probe = defaultdict(list)
mid = None
for line in f:
    if ':' in line:
        mid = int(line.split(':')[0])
    else:
        uid = int(line)
        probe[mid].append(uid)

w = open(root + 'probe_rated.txt', 'w')

for mid in MIDS:
    name = 'mv_{:0>7}.txt'.format(mid)

    if mid in probe:
        w.write('%d:\n' % mid)

    f = open(root + training_set + '/' + name)
    tf = open(root + 'noprobe/' + training_set + '/' + name, 'w')
    tf.write(f.readline())
    for line in f:
        u, r, _ = line.split(',')
        u = int(u)
        r = int(r)
        if u in probe[mid]:
            w.write(line)
        else:
            tf.write(line)


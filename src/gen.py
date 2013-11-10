"""
gen.py
Daniel Goldbach

Generates disjoint and nicely formatted training and test sets from the Netflix
challenge data. The test set contains data from Netflix's probe set. The
original challenge data combines all ratings into the training set, which
doesn't make sense to me. Better to have a clear separation between training and
test data.

"""

import argparse
import collections
import os

ROOT = '..'


def generate(mids, training_set):
    f = open(os.path.join(ROOT, 'probe.txt'))
    probe = collections.defaultdict(list)
    mid = None
    for line in f:
        if ':' in line:
            mid = int(line.split(':')[0])
        else:
            uid = int(line)
            probe[mid].append(uid)

    w = open(os.path.join(ROOT, 'probe_rated.txt'), 'w')

    for mid in mids:
        name = 'mv_{:0>7}.txt'.format(mid)

        if mid in probe:
            w.write('%d:\n' % mid)

        f = open(os.path.join(ROOT, training_set, name))
        tf = open(os.path.join(ROOT, 'no_probe_set/', name), 'w')
        tf.write(f.readline()) # first line contains movie ID then ':'
        for line in f:
            u, r, _ = line.split(',')
            u, r = int(u), int(r)

            # Write instance into either training set or test set based on type
            if u in probe[mid]:
                w.write(line)
            else:
                tf.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate formatted and appropriately sized training sets.')
    parser.add_argument(
        'num_movies',
        type=int,
        help='the number of movies to include in the datasets. 1 <= num_movies <= 17770.')
    parser.add_argument(
        '-f', '--fulldataset',
        action='store_true',
        help='use the full dataset as opposed to the medium dataset.')
    args = parser.parse_args()

    training_set_fn = 'training_set' if args.fulldataset else 'training_med'
    movie_ids = range(1, args.num_movies + 1)

    generate(movie_ids, training_set_fn)

    print 'successfully generated datasets.'

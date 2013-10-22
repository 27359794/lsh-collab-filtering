"""
training_med: U = ~15000 users
"""


import cPickle
import ipdb; bug = ipdb.set_trace
import itertools
import os
import prettytable
import pylab
import random
import scipy.sparse
import sklearn.metrics
import time

import numpy as np
import scipy.stats as stats

from collections import defaultdict

import cosine_nn

START_MID, END_MID = 1, 1000
MOVIE_IDS = range(START_MID, END_MID+1)
USER_IDS = None

WORK_DIR = '..'
TRAINING_SET = 'noprobe/training_med'
CACHE = 'cache'

UNCLUSTERED = -1
NOISE = -2


def main():
    # find_neighbours_for_each()
    index_by_users()

    # Require angle < 72deg
    # eps = 1-np.cos(3/7.0 * np.pi)# 1 - np.cos(2/5.0 * np.pi)
    # min_pts = 8
    # dbscan_parameter_est(min_pts)
    # dbscan(eps, min_pts)



def index_by_users():
    movie_vecs = get_movie_ratings()
    iindex = get_normalised_inverse_index(movie_vecs)

    movie_names = get_movie_names()
    user_ids = iindex.keys()

    print '{} users, RVF size is {}'.format(
            len(user_ids), len(user_ids)*cosine_nn.CosineNN.BLOCK_SIZE*cosine_nn.CosineNN.NUM_BLOCKS)

    nn_index = cosine_nn.CosineNN(END_MID + 1)  # +1 because mids are 1-based

    print 'NN setup complete. beginning indexing...'
    # Add all movies to nn_index
    for i, uid in enumerate(user_ids):
        col = iindex[uid]
        nn_index.index(uid, col)
        print i/float(len(user_ids)), uid
    print 'indexing and setup complete'

    t = prettytable.PrettyTable(['id', 'name', 'closest', 'dist'])

    uid = user_ids[0]
    eps = 1 - np.cos(1/2.0 * np.pi)
    print 'done'
    for i, uid in enumerate(user_ids):
        neighbours = nn_index.find_neighbours(uid, eps)
        print i, len(neighbours) / float(len(nn_index.query(uid)))
        by_dist = sorted(neighbours, key=lambda nuid: nn_index.cosine_dist_between(uid, nuid))
        t.add_row([
            uid,
            0,
            by_dist[0] if by_dist else '-',
            # dotdot(movie_names[mid]),
            # dotdot(movie_names[by_dist[0]]) if by_dist else '-',
            np.degrees(np.arccos(1 - nn_index.cosine_dist_between(by_dist[0], uid))) if by_dist else '-'
        ])
    bug()

    # for mid in MOVIE_IDS:
    #     # demand neighbours inside 72deg
    #     neighbours = nn_index.query(mid) - set([mid])
    #     by_dist = sorted(neighbours, key=lambda nmid: nn_index.cosine_dist_between(mid, nmid))
    #     t.add_row([
    #         mid,
    #         dotdot(movie_names[mid]),
    #         dotdot(movie_names[by_dist[0]]) if by_dist else '-',
    #         np.degrees(np.arccos(1 - nn_index.cosine_dist_between(by_dist[0], mid))) if by_dist else '-'
    #     ])
    print t


def find_neighbours_for_each():
    movie_vecs = get_user_normalised_ratings()
    movie_names = get_movie_names()
    viewer_set = {mi: set(movie_vecs[mi].keys()) for mi in MOVIE_IDS}
    user_ids = list(set.union(*viewer_set.itervalues()))

    min_degrees = 78.0
    eps = 1 - np.cos(np.radians(min_degrees))

    print '{} users, RVF size is {}'.format(
            len(user_ids), len(user_ids)*cosine_nn.CosineNN.BLOCK_SIZE*cosine_nn.CosineNN.NUM_BLOCKS)

    nn_index = cosine_nn.CosineNN(len(user_ids))

    print 'NN setup complete. beginning indexing...'
    # Add all movies to nn_index
    for mid in MOVIE_IDS:
        col = np.fromiter(
            ((movie_vecs[mid][u] if u in viewer_set[mid] else 0) for u in user_ids),
            int)
        nn_index.index(mid, col)
    print 'indexing and setup complete'

    t = prettytable.PrettyTable(['id', 'name', 'closest', 'dist'])

    for mid in MOVIE_IDS:
        # demand neighbours inside 72deg
        neighbours = nn_index.query(mid) - set([mid])
        by_dist = sorted(neighbours, key=lambda nmid: nn_index.cosine_dist_between(mid, nmid))
        t.add_row([
            mid,
            dotdot(movie_names[mid]),
            dotdot(movie_names[by_dist[0]]) if by_dist else '-',
            np.degrees(np.arccos(1 - nn_index.cosine_dist_between(by_dist[0], mid))) if by_dist else '-'
        ])
    print t

def dotdot(s):
    lim = 35
    return s if len(s)<lim else s[:lim-3] + '...'


def dbscan_parameter_est(min_pts):
    movie_vecs = get_user_normalised_ratings()
    movie_names = get_movie_names()
    viewer_set = {mi: set(movie_vecs[mi].keys()) for mi in MOVIE_IDS}
    user_ids = list(set.union(*viewer_set.itervalues()))
    visited = set()

    print 'beginning dbscan. {} users, RVF size is {}'.format(
            len(user_ids), len(user_ids)*cosine_nn.CosineNN.BLOCK_SIZE*cosine_nn.CosineNN.NUM_BLOCKS)

    nn_index = cosine_nn.CosineNN(len(user_ids))

    print 'NN setup complete. beginning indexing...'
    # Add all movies to nn_index
    for mid in MOVIE_IDS:
        col = np.fromiter(
            ((movie_vecs[mid][u] if u in viewer_set[mid] else 0) for u in user_ids),
            int)
        nn_index.index(mid, col)
    print 'indexing and setup complete'

    scatters = []

    for mid in MOVIE_IDS:
        # find 4-neighbourhood
        neighbours = nn_index.find_neighbours(mid, 999)  # eps=inf -- don't filter
        by_dist = []
        for nmid in neighbours:
            if nmid != mid:
                by_dist.append((nn_index.cosine_dist_between(mid, nmid), nmid))
        by_dist.sort()

        k = 8
        if len(by_dist) >= k:
            scatters.append(by_dist[k][0])
    scatters.sort(reverse=True)
    pylab.plot(scatters)
    pylab.show()


def dbscan(eps, min_pts):
    """
    for each unvisited point in dataset:
        mark as visited
        regionquery point
        if not enough points in region:
            mark as noise
        else:
            find cluster around point

    """
    movie_vecs = get_user_normalised_ratings()
    movie_names = get_movie_names()
    viewer_set = {mi: set(movie_vecs[mi].keys()) for mi in MOVIE_IDS}
    user_ids = list(set.union(*viewer_set.itervalues()))
    visited = set()

    print 'beginning dbscan. {} users, RVF size is {}'.format(
            len(user_ids), len(user_ids)*cosine_nn.CosineNN.BLOCK_SIZE*cosine_nn.CosineNN.NUM_BLOCKS)

    nn_index = cosine_nn.CosineNN(len(user_ids))

    cluster_ids = {mid: UNCLUSTERED for mid in MOVIE_IDS}
    cur_cluster_id = 0

    print 'NN setup complete. beginning indexing...'

    # Add all movies to nn_index
    for mid in MOVIE_IDS:
        col = np.fromiter(
            ((movie_vecs[mid][u] if u in viewer_set[mid] else 0) for u in user_ids),
            int)
        nn_index.index(mid, col)

    print 'indexing and setup complete'

    for mid in MOVIE_IDS:
        if mid in visited:
            continue
        visited.add(mid)

        neighbours = nn_index.find_neighbours(mid, eps)
        if len(neighbours) < min_pts:
            cluster_ids[mid] = NOISE
        else:
            expand_cluster(mid, neighbours, visited, nn_index,
                           cur_cluster_id, cluster_ids, eps, min_pts)
            cur_cluster_id += 1

    assert cluster_ids.values().count(UNCLUSTERED) == 0, 'all points visited'
    print 'NOISE:', cluster_ids.values().count(NOISE)
    print 'TOTAL', len(cluster_ids)

    for cluster_id in xrange(0, cur_cluster_id):
        print '-' * 100
        ms = {mid for (mid, cid) in cluster_ids.iteritems() if cid == cluster_id}
        for mid in ms:
            print movie_names[mid]
        cosine_pairs = [nn_index.cosine_dist_between(mida, midb)
                        for (mida, midb) in itertools.combinations(ms, 2)]
        print 'AVERAGE COSINE SIMILARITY OF CLUSTER MEMBERS:', np.mean(cosine_pairs)

    random_mids = random.sample(MOVIE_IDS, min(len(MOVIE_IDS), 100))  # Without replacement
    cosine_pairs = [nn_index.cosine_dist_between(mida, midb)
                    for (mida, midb) in itertools.combinations(random_mids, 2)]

    cosine_pairs = np.compress(~np.isnan(cosine_pairs), cosine_pairs)
    print 'AVERAGE COSINE SIMILARITY OF CONTROL GROUP:', stats.nanmean(cosine_pairs)



def expand_cluster(mid, neighbours, visited, nn_index,
                   cur_cluster_id, cluster_ids, eps, min_pts):
    cluster_ids[mid] = cur_cluster_id
    stack = neighbours  # this is a set
    while stack:
        cur_pt = stack.pop()  # cur_pt is a movie ID
        if cur_pt not in visited:
            visited.add(cur_pt)
            next_neighbours = nn_index.find_neighbours(cur_pt, eps)
            if len(next_neighbours) >= min_pts:
                stack.update(next_neighbours)
        if cluster_ids[cur_pt] in [UNCLUSTERED, NOISE]:
            print 'assigning to', cur_cluster_id
            cluster_ids[cur_pt] = cur_cluster_id


def cached(f):
    name = '{}_{}-{}_{}_cache.pkl'.format(f.__name__, START_MID,
                                           END_MID, TRAINING_SET)
    name = name.replace('/', '_slash_')
    fpath = os.path.join(WORK_DIR, CACHE, name)
    def writeToCache(data):
        with open(fpath, 'wb') as fout:
            cPickle.dump(data, fout)
        return data

    def helper(*args):
        assert not args  # This doesn't cache funcs with arguments

        if not os.path.exists(fpath):
            return writeToCache(f())
        else:
            try:
                print 'reading', f.__name__, 'from cache'
                return cPickle.load(open(fpath, 'rb'))
            except EOFError:
                return writeToCache(f())

    return helper


@cached
def read_probe():
    f = open(os.path.join(WORK_DIR, 'probe.txt'))
    li = set()
    d = {}
    lastid = None
    for line in f:
        if ':' in line:
            lastid = int(line.strip()[:-1])
            d[lastid] = li
            li = set()
        else:
            li.add(int(line))
    return d


def fuzz_plot(xs, ys):
    pylab.scatter([x+random.random()-0.5 for x in xs],
                  [y+random.random()-0.5 for y in ys])
    pylab.show()


@cached
def get_user_normalised_ratings():
    """
    Standardise each user's rating by subtracting their mean, dividing by
    their stddev.

    """
    ratings = get_movie_ratings()
    print 'read raw ratings...'
    iindex = inverse_index(ratings)

    pylab.hist(map(len, iindex.values()))
    pylab.show()
    print 'built inverse index...'
    user_stats = {}  # userid -> (mean, stddev)
    for u, uvec in iindex.iteritems():
        user_ratings = [r for (mi, r) in uvec]
        user_stats[u] = np.mean(user_ratings), np.std(user_ratings)

    for mi in MOVIE_IDS:
        mvec = ratings[mi]
        for u in mvec:
            umean, ustddev = user_stats[u]
            mvec[u] = (mvec[u] - umean) / (ustddev if ustddev > 0 else 1)

    print 'normalised ratings...'
    return ratings


def get_normalised_inverse_index(movie_ratings):
    """
    A dict of userid : sparse.csr_matrix]. The ith element of the sparse row is the
    user's normalised rating for the ith movie in MOVIE_IDS or 0 if they didn't
    see that movie.

    """
    user_index = defaultdict(list)

    assert MOVIE_IDS == sorted(MOVIE_IDS)
    for mi in MOVIE_IDS:
        for (uid, rating) in movie_ratings[mi].iteritems():
            user_index[uid].append((mi, rating))

    iindex = {}

    for uid in user_index:
        uratings = [r for (m, r) in user_index[uid]]
        umean, ustddev = np.mean(uratings), np.std(uratings)
        if ustddev == 0:
            ustddev = 1  # Prevent div by 0

        vec = [0] * (END_MID+1)#np.array(user_index[uid].todense())
        for (mi, r) in user_index[uid]:
            vec[mi] = (r-umean) / ustddev if r > 0 else 0
        sparsified = scipy.sparse.csc_matrix(vec).T

        if (len(set(uratings)) > 1 and
            len(uratings) > 10):  # if our sparsified vector is interesting
            iindex[uid] = sparsified

    return iindex


def inverse_index(movie_ratings):
    """A dict of userid : [(movie id, rating), ...]"""
    user_index = defaultdict(list)
    for mi in MOVIE_IDS:
        for u, rating in movie_ratings[mi].iteritems():
            user_index[u].append((mi, rating))
    return user_index



@cached
def get_movie_ratings():
    """A list whose ith element is a dict {user: rating} for movie i."""
    movie_filenames = [(mi, 'mv_{:0>7}.txt'.format(mi)) for mi in MOVIE_IDS]
    # No movie with id 0, leave that index blank
    movie_vectors = [-1337] + [{} for _ in range(END_MID+1)]
    for i, fn in movie_filenames:
        f = open(os.path.join(WORK_DIR, TRAINING_SET, fn))
        # fo = open('training_med/' + fn, 'w')
        # fo.write(f.readline())  # Ignore first line, which contains "id:"
        f.readline()
        d = {}

        for line in f:
            uid, rating, date = line.split(',')
            d[int(uid)] = int(rating)
        f.close()
        movie_vectors[i] = d
    print 'loaded ratings from training data'
    return movie_vectors

@cached
def get_movie_names():
    """A dictionary of movie id to movie name."""
    movie_names = {}
    for line in open(os.path.join(WORK_DIR, 'movie_titles.txt')):
        i, year, name = line.strip().split(',', 2)
        movie_names[int(i)] = name
    return movie_names

if __name__ == '__main__':
    import doctest; doctest.testmod()
    main()

"""
code.py
Author: Daniel Goldbach

Run collaborative filtering on Netflix challenge dataset and output RMSE.
"""

import cPickle
import numpy as np
import os
import scipy.sparse
from collections import defaultdict

import cosine_nn

START_MID, END_MID = 1, 1000
MOVIE_IDS = range(START_MID, END_MID+1)
USER_IDS = None

WORK_DIR = '..'
TRAINING_SET = 'noprobe/training_med'
RATED_PROBE_FN = 'probe_rated.txt'
CACHE = os.path.join(WORK_DIR, 'cache')

UNCLUSTERED = -1
NOISE = -2


def main():
    index_by_users()


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
        col = iindex[uid].uratings
        nn_index.index(uid, col)
        if i % 10 == 0:
            print i/float(len(user_ids)), uid
    print 'indexing and setup complete'

    probe_ratings = read_probe()

    to_predict = []
    for mid, mratings in probe_ratings.iteritems():
        for uid, rating in mratings.iteritems():
            if uid in user_ids:
                to_predict.append((uid, mid, rating))

    errors = []
    for i, (uid, mid, actual) in enumerate(to_predict):
        print i / float(len(to_predict))
        guess = guess_rating(nn_index, iindex, uid, mid)
        errors.append((guess - actual)**2)
    print 'RMSE:', np.sqrt(np.mean(errors))


def guess_rating(nn_index, iindex, uid, mid):
    """Guess NORMALISED rating. You should scale this by uid's mean and std."""
    eps = 1 - np.cos(1/2.0 * np.pi)
    neighbours = nn_index.find_neighbours(uid, eps)
    num_neighbours_considered = 0
    sum_inv_cosdist = 0
    weighted_mean = 0
    wasted = 0
    for (nuid, cosdist) in neighbours:
        assert nuid != uid
        if cosdist > eps:
            wasted += 1
        neighbour_rating = iindex[nuid].uratings[mid, 0]
        if neighbour_rating != 0:  # Ignore non-ratings
            # Scales contribution of closer neighbours higher.
            # e.g. if there are only two neighbours of distance 60deg, 90deg,
            # the 60deg neighbour contributes 66%, the 90deg contributes 33%
            num_neighbours_considered += 1
            weighted_mean += neighbour_rating / cosdist
            sum_inv_cosdist += 1/cosdist
    print 'before div', weighted_mean, sum_inv_cosdist
    weighted_mean /= (sum_inv_cosdist if sum_inv_cosdist else 1)
    print 'weighted mean', weighted_mean

    # # Now calculate the guess based on the neighbour average, uid's mean and std
    # Order of *, + MATTERS here! Opposite order of normalisation
    guess = weighted_mean
    guess *= iindex[uid].std
    guess += iindex[uid].mean

    # Put it in the actual range of possible ratings (e.g. don't guess -1)
    guess = min(max(guess, 1), 5)
    print ('guessing', guess, 'based on', num_neighbours_considered, 'of',
        wasted+num_neighbours_considered, 'mean is', iindex[uid].mean,
        'std', iindex[uid].std)
    return guess


def cached(f):
    name = '{}_{}-{}_{}_cache.pkl'.format(f.__name__, START_MID,
                                           END_MID, TRAINING_SET)
    name = name.replace('/', '_slash_')
    fpath = os.path.join(CACHE, name)
    def writeToCache(data):
        with open(fpath, 'wb') as fout:
            cPickle.dump(data, fout)
        return data

    def helper(*args):
        if args:
            print "WARNING: probably shouldn't use @cached on funcs with args!"
        if not os.path.exists(fpath):
            return writeToCache(f(*args))
        else:
            try:
                print 'reading', f.__name__, 'from cache'
                return cPickle.load(open(fpath, 'rb'))
            except EOFError:
                return writeToCache(f(*args))

    return helper


@cached
def read_probe():
    """A list whose ith element is a dict {user: rating} for movie i."""
    f = open(os.path.join(WORK_DIR, RATED_PROBE_FN))
    movie_vectors = {}
    d = {}
    lastid = None
    for line in f:
        if ':' in line:
            lastid = int(line.strip()[:-1])
            if lastid >= START_MID and lastid <= END_MID:
                movie_vectors[lastid] = {}
        else:
            uid, rating, date = line.split(',')
            if lastid >= START_MID and lastid <= END_MID:
                movie_vectors[lastid][int(uid)] = int(rating)
    return movie_vectors


@cached
def get_user_normalised_ratings():
    """
    Standardise each user's rating by subtracting their mean, dividing by
    their stddev.

    """
    ratings = get_movie_ratings()
    print 'read raw ratings...'
    iindex = inverse_index(ratings)

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


class InverseIndexEntry(object):
    # __slots__ = ('mean', 'std', 'uratings')

    def __init__(self, mean, std, uratings):
        self.mean = mean
        self.std = std
        self.uratings = uratings

@cached
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
            iindex[uid] = InverseIndexEntry(umean, ustddev, sparsified)

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
    main()

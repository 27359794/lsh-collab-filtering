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

WORK_DIR = '..'
TRAINING_SET = 'noprobe/training_med'
RATED_PROBE_FN = 'probe_rated.txt'
CACHE = os.path.join(WORK_DIR, 'cache')


def main():
    index_by_users()


class InverseIndexEntry(object):
    __slots__ = ('mean', 'std', 'uratings')

    def __init__(self, mean, std, uratings):
        self.mean = mean
        self.std = std
        self.uratings = uratings


def index_by_users():
    iindex = get_normalised_inverse_index(get_movie_ratings())
    user_ids = iindex.keys()
    nn_index = cosine_nn.CosineNN(END_MID + 1)  # +1 because mids are 1-based

    # Add all movies to nn_index
    for i, uid in enumerate(user_ids):
        col = iindex[uid].uratings
        nn_index.index(uid, col)
        if i % 10 == 0:
            print i/float(len(user_ids)), uid
    print 'indexing and setup complete'

    # Find all test data instances for which we can predict a rating
    probe_ratings = read_probe()
    to_predict = []
    for mid, mratings in probe_ratings.iteritems():
        for uid, rating in mratings.iteritems():
            if uid in user_ids:
                to_predict.append((uid, mid, rating))

    # Predict ratings for test data instances and add up error
    errors = []
    for i, (uid, mid, actual) in enumerate(to_predict):
        print 'progress:', i / float(len(to_predict))
        guess = guess_rating(nn_index, iindex, uid, mid)
        errors.append((guess - actual)**2)
    print 'RMSE:', np.sqrt(np.mean(errors))


def guess_rating(nn_index, iindex, uid, mid):
    """Guess NORMALISED rating. You should scale this by uid's mean and std."""

    # This is the threshold for neighbours. If neighbour distance is further
    # than this, they're not a neighbour so ignore them.
    threshold_dist = 1 - np.cos(1/2.0 * np.pi)
    neighbours = nn_index.query_with_dist(uid, threshold_dist)
    sum_inv_cosdist = 0
    weighted_mean = 0
    for (nuid, cosdist) in neighbours:
        assert nuid != uid, "can't be a neighbour of itself"
        neighbour_rating = iindex[nuid].uratings[mid, 0]

        # Ignore non-ratings and non-neighbours.
        if neighbour_rating != 0 and cosdist < threshold_dist:
            # Scales contribution of closer neighbours higher.
            # e.g. if there are only two neighbours of distance 60deg, 90deg,
            # the 60deg neighbour contributes 66%, the 90deg contributes 33%
            weighted_mean += neighbour_rating / cosdist
            sum_inv_cosdist += 1 / cosdist

    weighted_mean /= (sum_inv_cosdist if sum_inv_cosdist else 1)

    # # Now calculate the guess based on the neighbour average, uid's mean/std
    guess = (weighted_mean * iindex[uid].std) + iindex[uid].mean

    # Put it in the actual range of possible ratings (e.g. don't guess -1)
    guess = min(max(guess, 1), 5)

    return guess


def cached(f):
    name = '{}_{}-{}_{}_cache.pkl'.format(f.__name__, START_MID,
                                          END_MID, TRAINING_SET)
    name = name.replace('/', '_slash_')  # Filenames with slashes break stuff
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
                # Cached version is corrupt
                return writeToCache(f(*args))

    return helper


@cached
def read_probe():
    """A list whose ith element is a dict {user: rating} for movie i."""
    f = open(os.path.join(WORK_DIR, RATED_PROBE_FN))
    movie_vectors = {}
    lastid = None
    for line in f:
        if ':' in line:
            lastid = int(line.strip()[:-1])  # Strip off colon
            if lastid >= START_MID and lastid <= END_MID:
                movie_vectors[lastid] = {}
        else:
            uid, rating, date = line.split(',')
            assert lastid is not None, 'first line of probe must be movie id'
            if lastid >= START_MID and lastid <= END_MID:
                movie_vectors[lastid][int(uid)] = int(rating)
    return movie_vectors


@cached
def get_normalised_inverse_index(movie_ratings):
    """
    A dict of userid : sparse.csr_matrix]. The ith element of the sparse row is
    the user's normalised rating for the ith movie in MOVIE_IDS or 0 if they
    didn't see that movie.

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

        vec = [0] * (END_MID+1)
        for (mi, r) in user_index[uid]:
            # Don't try to normalise nil ratings, leave them as 0
            vec[mi] = (r-umean) / ustddev if r > 0 else 0
        sparsified = scipy.sparse.csc_matrix(vec).T

        if (len(set(uratings)) > 1 and
            len(uratings) > 10):  # If our sparsified vector is interesting
            iindex[uid] = InverseIndexEntry(umean, ustddev, sparsified)

    return iindex


@cached
def get_movie_ratings():
    """A list whose ith element is a dict {user: rating} for movie i."""
    movie_filenames = [(mi, 'mv_{:0>7}.txt'.format(mi)) for mi in MOVIE_IDS]
    movie_vectors = {}
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

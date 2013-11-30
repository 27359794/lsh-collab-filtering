"""
main.py
Daniel Goldbach
2013 October

Run collaborative filtering on Netflix challenge dataset and output RMSE.

For quicker debugging time, uncomment the @cached decorations above functions
that can be cached.
"""

import argparse
import cPickle
import numpy as np
import os
import scipy.sparse
from collections import defaultdict

# Project-specific imports
import cosine_nn
import utils


WORK_DIR = '..'
TRAINING_SET = 'no_probe_set/'
RATED_PROBE_FN = 'probe_rated.txt'
CACHE = os.path.join(WORK_DIR, 'cache')


def main():
    parser = argparse.ArgumentParser(
        description='Run CF and output RMSE.')
    parser.add_argument(
        'num_movies',
        type=int,
        help='the number of movies in the datasets to process. '
             '1 <= num_movies <= 17770.')
    arguments = parser.parse_args()

    start_mid, end_mid = 1, arguments.num_movies
    index_and_evaluate(set(range(start_mid, end_mid)))


class InverseIndexEntry(object):
    """
    Named tuple for a user's inverse index entry. Contains a list of their
    ratings for each movie, as well as their mean rating and the std deviation
    of their ratings.

    TODO: use actual named tuple
    """
    def __init__(self, mean, std, uratings):
        self.mean = mean
        self.std = std
        self.uratings = uratings


def index_and_evaluate(movie_ids):
    iindex = get_normalised_inverse_index(movie_ids)
    user_ids = iindex.keys()
    nn_index = cosine_nn.CosineNN(max(movie_ids) + 1) # +1 since ids are 1-based

    # Add all movies to nn_index
    for i, uid in enumerate(user_ids):
        col = iindex[uid].uratings
        nn_index.index(uid, col)
        if i % 10 == 0:
            print 'index progress: {:.3%}'.format(i / float(len(user_ids)))
    print 'indexing and setup complete'

    # Find all test data instances for which we can predict a rating
    probe_ratings = read_probe(movie_ids)
    to_predict = []
    for mid, mratings in probe_ratings.iteritems():
        for uid, rating in mratings.iteritems():
            if uid in user_ids:
                to_predict.append((uid, mid, rating))

    # Predict ratings for test data instances and add up error
    errors = []
    for i, (uid, mid, actual) in enumerate(to_predict):
        print 'evaluation progress: {:.3%}'.format(i / float(len(to_predict)))
        guess = guess_rating(nn_index, iindex, uid, mid)
        errors.append((guess - actual)**2)
    print 'RMSE:', np.sqrt(np.mean(errors))


def guess_rating(nn_index, iindex, uid, mid):
    """Guess NORMALISED rating. You should scale this by uid's mean and std."""
    # This is the threshold for neighbours. If neighbour distance is further
    # than this, they're not a neighbour so ignore them.
    threshold_dist = 1 - np.cos(1/2.0 * np.pi)
    neighbours = nn_index.query_with_dist(uid)
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
    """Decorator to cache function return values to disk.

    Note that this does NOT care about function arguments! So f(2) and f(1) will
    look the same to the cache. So don't cache a function with arguments unless
    you know exactly what you're doing.
    """
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
                # Current cached file is corrupt. Ignore it, generate a new one.
                return writeToCache(f(*args))

    return helper


# @cached
def read_probe(movie_ids):
    """A list whose ith element is a dict {user: rating} for movie i."""
    probe_file = open(os.path.join(WORK_DIR, RATED_PROBE_FN))
    movie_vectors = {}
    lastid = None
    for line in probe_file:
        if ':' in line:
            lastid = int(line.strip()[:-1])  # Strip off colon
            if lastid in movie_ids:
                movie_vectors[lastid] = {}
        else:
            uid, rating, _ = line.split(',')
            assert lastid is not None, 'first line of probe must be movie id'
            if lastid in movie_ids:
                movie_vectors[lastid][int(uid)] = int(rating)
    return movie_vectors


# @cached
def get_normalised_inverse_index(movie_ids):
    """
    A dict of userid : sparse.csr_matrix]. The ith element of the sparse row is
    the user's normalised rating for the ith movie in MOVIE_IDS or 0 if they
    didn't see that movie.

    @param movie_ratings get_movie_ratings(movie_ids)
    """
    movie_ratings = get_movie_ratings(movie_ids)
    sorted_movie_ids = sorted(movie_ids)
    iindex = {}

    # Get non-normalised inverse index
    user_index = defaultdict(list)
    for mid in sorted_movie_ids:
        for (uid, rating) in movie_ratings[mid].iteritems():
            user_index[uid].append((mid, rating))

    for uid, udata in user_index.iteritems():
        uratings = [r for (mid, r) in udata]
        umean, ustddev = np.mean(uratings), np.std(uratings)

        # If user's ratings are uninteresting (not enough ratings or not enough
        # distinct ratings), don't index user
        if len(uratings) <= 10 or ustddev == 0:
            continue

        # user_vec[mid] is uid's normalised rating for mid
        user_vec = [0] * (max(sorted_movie_ids) + 1)
        for (mid, rating) in user_index[uid]:
            if rating != 0:  # Leave lack of rating as 0
                user_vec[mid] = (rating - umean) / ustddev
        sparsified = scipy.sparse.csc_matrix(user_vec).T
        iindex[uid] = InverseIndexEntry(umean, ustddev, sparsified)

    return iindex


# @cached
def get_movie_ratings(movie_ids):
    """@return {movieid : {userid: rating}"""
    movie_filenames = [(mi, 'mv_{:0>7}.txt'.format(mi)) for mi in movie_ids]
    movie_vectors = {}
    for i, fn in movie_filenames:
        assert os.path.exists(os.path.join(WORK_DIR, TRAINING_SET, fn)), \
               'ensure dataset containing {} has been generated'.format(fn)
        movie_file = open(os.path.join(WORK_DIR, TRAINING_SET, fn))
        movie_file.readline()

        uid_to_rating = {}
        for line in movie_file:
            uid, rating, date = line.split(',')
            uid_to_rating[int(uid)] = int(rating)
        movie_vectors[i] = uid_to_rating
        movie_file.close()
    print 'loaded ratings from training data'
    return movie_vectors


# @cached
def get_movie_names():
    """A dictionary of movie id to movie name."""
    movie_names = {}
    for line in open(os.path.join(WORK_DIR, 'movie_titles.txt')):
        mid, year, name = line.strip().split(',', 2)
        movie_names[int(mid)] = name
    return movie_names


if __name__ == '__main__':
    main()

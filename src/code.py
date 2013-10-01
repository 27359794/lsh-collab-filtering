"""
training_med: U = 11804 users
"""

import scipy.stats as stats
import sklearn.metrics
import random
import time
import cPickle
import prettytable
import pylab
import numpy as np
from collections import defaultdict
import ipdb; bug = ipdb.set_trace
import os

import utils
import cosine_nn

START_MID, END_MID = 1, 300
MOVIE_IDS = range(START_MID, END_MID+1)
USER_IDS = None

WORK_DIR = '..'
TRAINING_SET = 'training_med'
CACHE = 'cache'





def main():
    ## INPUT AND SETUP
    movie_vecs = get_user_normalised_ratings()
    movie_names = get_movie_names()
    viewer_set = {mi: set(movie_vecs[mi].keys()) for mi in MOVIE_IDS}
    user_list = set()
    for v in viewer_set.itervalues():
        user_list.update(v)
    user_list = list(user_list)
    USER_IDS = user_list
    print 'generated viewer sets...'

    # most_popular_id = np.argmax(map(len, movie_vecs[1:])) + 1

    nn = cosine_nn.CosineNN(len(user_list))

    # print 'I AM', movie_names[most_popular_id]



# def dbscan(points):



def cached(f):
    name = '{}_{}-{}_{}_cache.pkl'.format(f.__name__, START_MID,
                                           END_MID, TRAINING_SET)
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
    f = open('probe.txt')
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
    """Normalise each user's rating by subtracting their mean rating."""
    ratings = get_movie_ratings()
    print 'read raw ratings...'
    iindex = inverse_index(ratings)
    print 'built inverse index...'
    user_mean = {}
    for u, uvec in iindex.iteritems():
        user_mean[u] = np.mean([r for (mi, r) in uvec])

    for mi in MOVIE_IDS:
        mvec = ratings[mi]
        for u in mvec:
            mvec[u] -= user_mean[u]

    print 'normalised ratings...'
    return ratings


# def inverse_index(movie_ratings):
#     """A dict of userid : [(movie id, rating), ...]"""
#     user_index = defaultdict(list)
#     for mi in MOVIE_IDS:
#         # import pdb; pdb.set_trace()
#         for u, rating in movie_ratings[mi].iteritems():
#             user_index[u].append((mi, rating))
#     return user_index


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

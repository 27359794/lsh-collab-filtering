"""
training_med: U = 11804 users
"""

import scipy.stats as stats
import scipy.spatial as spatial
import random
import cPickle
import prettytable
import pylab
import numpy as np
from collections import defaultdict
import ipdb; bug = ipdb.set_trace
import os

START_MID, END_MID = 1, 300
MOVIE_IDS = range(START_MID, END_MID+1)
USER_IDS = None

RANDOM_VECTOR_FAMILY = None

WORK_DIR = '..'

TRAINING_SET = 'training_med'
CACHE = 'cache'

BLOCK_SIZE = 9
NUM_BLOCKS = 65
SIG_LENGTH = BLOCK_SIZE * NUM_BLOCKS


def main():
    ## INPUT AND SETUP
    movie_vecs = get_movie_ratings()
    movie_names = get_movie_names()
    
    viewer_set = {mi: set(movie_vecs[mi].keys()) for mi in MOVIE_IDS}
    user_list = set()
    for v in viewer_set.itervalues():
        user_list.update(v)
    user_list = list(user_list)
    global USER_LIST, RANDOM_VECTOR_FAMILY
    USER_IDS = list(user_list)
    RANDOM_VECTOR_FAMILY = np.random.rand(SIG_LENGTH, len(USER_IDS)) - 0.5
    print 'generated viewer sets...'

    most_popular_id = np.argmax(map(len, movie_vecs[1:])) + 1

    ## GENERATE SIGNATURES

    full_vecs = {}

    signatures = {}
    for m in MOVIE_IDS:
        # Form the column vector of the utility matrix for m
        col = np.fromiter(
                ((movie_vecs[m][u] if u in viewer_set[m] else 0)
                 for u in USER_IDS),
                int)
        signatures[m] = signature_of(col)
        full_vecs[m] = col
    

    # nn_index[(which_block, block_integer_value)] = list of m-ids in bucket
    nn_index = defaultdict(list)

    for mid, sig in signatures.iteritems():
        # For each block of BLOCK_SIZE bits, starting at the right, extract it
        for block_num in xrange(NUM_BLOCKS):
            # Shift the sig right until the last BLOCK_SIZE bits are the block,
            # Then take off the bits before the block.
            block_val = extract_block(sig, block_num)
            nn_index[(block_num, block_val)].append(mid)

    """
    Let's do some maths.

    m is a neighbour of n if the angle between m and n is less than a.

    We are querying for neighbours of n. m is an arbitrary element in the space
    S. Let the result set R := query(n). We want to know:
    - Precision (if m in R, how likely is it that m is a neighbour?)
    - Recall (if m is a neighbour, how likely is it that m in R?)

    P(m is neighbour) = a/pi
    P(m in R) = P(any block in m matches the corresponding block in n)
    P(a given block matches) = (1-a/pi)^BLOCK_SIZE
    P(m in resultset | m is neighbour)
            = P(any block in m matches) 
            = 1 - P(no blocks match)
            = 1 - (1 - P(given block matches))^NUM_BLOCKS
            = 1 - (1 - (1-a/pi)^BLOCK_SIZE)^NUM_BLOCKS

    Initial LSH has sensitivity:
    (pi/3, pi/2, 1-1/3, 1-1/2) = (pi/3, pi/2, .66, .50)
    Applying blocks of 4-bits, we get
    (pi/3, pi/2, (1-1/3)^4, (1-1/2)^4) = (pi/3, pi/2, .20, 0.06)
    Applying a union over 6 blocks, we get
    (pi/3, pi/2, 1-(1-(1-1/3)^4)^6, 1-(1-(1-1/2)^4)^6) = (.., .73, 0.32)

    Using block size 9 with 65 blocks, we get good values
    In [31]: 1-(1-(1-1/3.0)**9)**65, 1-(1-(1-1/2.0)**9)**65
    Out[31]: (0.819708249125523, 0.11933437406869052)
    i.e. a (pi/3, pi/2, 0.82, 0.12)-sensitive hash family.
    There's a >=0.82 chance of neighbours being in the result set,
    and a <=0.12 chance of non-neighbours being in the result set.
    """
    def query(mid):
        sig = signatures[mid]
        resultset = set()
        for block_num in xrange(NUM_BLOCKS):
            block_val = extract_block(sig, block_num)
            resultset.update(nn_index[(block_num, block_val)])
        return list(resultset)

    close = query(most_popular_id)
    for mid in MOVIE_IDS:
        actual_angle = np.arccos(1 - cos_dist(full_vecs[mid],
                                            full_vecs[most_popular_id]))
        if actual_angle < np.pi/3:
            print 'ACTUALLY NEIGHBOUR:', movie_names[mid]

    for mid in close:
        actual_angle = np.arccos(1 - cos_dist(full_vecs[mid],
                                            full_vecs[most_popular_id]))
        print movie_names[mid], np.degrees(actual_angle)



def cos_dist(vec1, vec2):
    """Cosine distance, 1-cos(theta)."""
    return spatial.distance.cosine(vec1, vec2)


def extract_block(sig, block_num):
    # Shift the sig right until the last BLOCK_SIZE bits are the block,
    # Then take off the bits before the block.
    return (sig >> (block_num*BLOCK_SIZE)) % (1 << BLOCK_SIZE)

def signature_of(vec):
    """Takes a numpy vector of length U and produces its LSH."""
    sketch = RANDOM_VECTOR_FAMILY.dot(vec)
    num = 0
    counter = 0
    for k in sketch:
        if k >= 0:
            num |= (1 << counter)
        counter += 1
    return num


def nxor_count_bits(a, b):
    """
    >>> nxor_count_bits(0b1011, 0b1010) 
    1
    >>> nxor_count_bits(0b10, 0b1)
    2
    >>> nxor_count_bits(0b10101010, 0b01010101)
    8
    """ 
    x = 1
    num_set = 0
    for i in xrange(SIG_LENGTH):
        num_set += ((a&x)>0) != ((b&x)>0)
        x <<= 1
    return num_set


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
        user_mean[u] = np.mean([r for (mi,r) in uvec])

    for mi in MOVIE_IDS:
        mvec = ratings[mi]
        for u in mvec:
            mvec[u] -= user_mean[u]

    print 'normalised ratings...'
    return ratings


def inverse_index(movie_ratings):
    """A dict of userid : [(movie id, rating), ...]"""
    user_index = defaultdict(list)
    for mi in MOVIE_IDS:
        # import pdb; pdb.set_trace()
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
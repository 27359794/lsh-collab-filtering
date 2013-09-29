"""
100m ratings
18000 movies
480000 users

on avg 5010 users per movie_vectors

most movies do not have 'nearest neighbours'. kill bill and kill bill v2 are
neighbours, but a movie like shawshank redemption doesn't necessarily have any
nearest neighbours.

the intersection of the audiences for a number of movies becomes very small
very fast. breaking these up into 2^basis_size during multi_corr makes them
tiny.

because intersections of audiences are so small, all pairs of vectors are
near-orthogonal. the way to fix this would be to remove the division by
vector norms -- but then it's not cosine anymore, and LSH fails.

training_small: 2.5% of full
training_med: 4.8% of full

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
FAMILY_SIZE = 300

TRAINING_SET = 'training_med'


def complete():
    movie_vecs = get_movie_ratings()
    movie_names = get_movie_names()
    movies = MOVIE_IDS[:]  # gonna get sorted
    
    viewer_set = {mi: set(movie_vecs[mi].keys()) for mi in MOVIE_IDS}
    viewer_list = set()
    for v in viewer_set.itervalues():
        viewer_list.update(v)
    viewer_list = list(viewer_list)
    print 'generated viewer sets...'

    islands = defaultdict(set)

    # Start with the most popular movie as our basis.
    movies.sort(key=lambda mi: len(movie_vecs[mi]), reverse=True)
    # basis = [11064]
    # assert START_MID <= basis[0] <= END_MID, 'basis movie is a valid movie id'
    # islands[basis[0]].add(basis[0])
    # basis_audience = reduce(set.intersection,
    #                         (set(movie_vecs[bi].keys()) for bi in basis))
    
    mt = prettytable.PrettyTable(['a', 'b', 'dist'])
    mt.sortby = 'dist'
    
    bla = []
    signatures = {}
    for o in xrange(len(movies)):
        oi = movies[o]
        watched = [movie_vecs[oi][v] if v in viewer_set[oi] else 0 for v in viewer_list]
        sig = signature_of(watched)
        signatures[oi] = sig
        bla.append(bin(sig)[2:].zfill(300))
        # print 'signed', o
    print 'generated sigs'
    bug()

    data = []
    tot = 0
    for a in xrange(len(movies)):
        print 'beginning', a
        ai = movies[a]
        # mt = prettytable.PrettyTable(['dist', 'ma', 'mb'])
        # mt.sortby = 'dist'
        # if a % 10 == 0:
        #     for tup in sorted(bla)[:200]:
        #         mt.add_row(tup)
        #     print mt[:100]
        siga = signatures[ai]
        watcheda = [movie_vecs[ai][v] if v in viewer_set[ai] else 0 for v in viewer_list]
        for b in xrange(a+1, len(movies)):
            bi = movies[b]
            
            watchedb = [movie_vecs[bi][v] if v in viewer_set[bi] else 0 for v in viewer_list]

            actual = np.degrees(np.arccos(1-spatial.distance.cosine(watcheda, watchedb)))
            # if np.isnan(actual):
            #     bug()
            val = nxor(siga, signatures[bi])
            theta_estimate = 180*val/float(FAMILY_SIZE)
            d = (actual - theta_estimate)**2
            if not np.isnan(actual):
                tot += d
            # print theta_estimate

            data.append((theta_estimate, ai, bi))
        if a>10:
            break
    # cPickle.dump(data, open('cosines.pkl', 'wb'))
    print np.sqrt(tot/float(len(MOVIE_IDS)))
    bug()

'''

def find_basis():
    # get_movie_ratings()
    # return
    # probe = read_probe()
    
    movie_vecs = get_movie_ratings()#get_user_normalised_ratings()#get_movie_ratings()
    movie_names = get_movie_names()
    movies = MOVIE_IDS[:]  # gonna get sorted
    # print 'wtf'
    viewer_set = {mi: set(movie_vecs[mi].keys()) for mi in MOVIE_IDS}
    viewer_list = set()
    for v in viewer_set.itervalues():
        viewer_list.update(v)
    viewer_list = list(viewer_list)
    print 'generated viewer sets...'    

    p = prettytable.PrettyTable([
        'name',
        '# movie viewers',
        '# movies in island',
        '# viewers in island'
    ])

    islands = defaultdict(set)

    # Start with the most popular movie as our basis.
    movies.sort(key=lambda mi: len(movie_vecs[mi]), reverse=True)

    basis = [11064]
    assert START_MID <= basis[0] <= END_MID, 'basis movie is a valid movie id'
    islands[basis[0]].add(basis[0])
    basis_audience = reduce(set.intersection,
                            (set(movie_vecs[bi].keys()) for bi in basis))
    
    mt = prettytable.PrettyTable(['movie', 'cosine', 'estimate'])
    mt.sortby = 'estimate'
    
    signatures = {}
    for o in xrange(len(movies)):
        oi = movies[o]
        watched = [movie_vecs[oi][v] if v in viewer_set[oi] else 0 for v in viewer_list]
        sig = signature_of(watched)
        signatures[oi] = sig
        # print 'signed', o
    print 'generated sigs'

    bi = basis[0]
    # joint_bi = [movie_vecs[bi][v] if v in viewer_set[bi] else 0 for v in viewer_list]
    sigb = signatures[bi]#signature_of(joint_bi)
    
    for o in xrange(len(movies)):
        oi = movies[o]
        
        # for bi in basis:
        who_saw_both = viewer_set[bi].intersection(viewer_set[oi])
        if len(who_saw_both) < 3:
            continue
        
        # joint_oi = [movie_vecs[oi][v] if v in viewer_set[oi] else 0 for v in viewer_list]
        sigo = signatures[oi]
        
        # corr, pv = stats.pearsonr(joint_bi, joint_oi)
        
        # dist = spatial.distance.cosine(joint_bi, joint_oi)
        val = nxor(sigb, sigo)
        
        # mt.add_row([movie_names[oi][:30], 180*np.arccos(1-dist)/np.pi, 180*val/float(FAMILY_SIZE)])
        mt.add_row([movie_names[oi][:30], 0, 180*val/float(FAMILY_SIZE)])
    print mt

    print 'basis', movie_names[basis[0]]
    
    bug()
    return



    # Next, we add to the basis a movie m which
    #  - has a significant amount of audience overlap with all of the basis
    #    (heuristic: is popular enough).
    #  - "is not in the span of the basis" (is not predictable by our basis
    #    / likes and dislikes on m cannot be predicted by observing likes
    #    and dislikes on the basis movies).
    # Invariant: all movie ids <= max(basis) have been considered already.
    for o in xrange(len(movies)):
        print 'processing', o
        oi = movies[o]

        joint_viewers = basis_audience.intersection(movie_vecs[oi].keys())

        if len(joint_viewers) <= 2:
            # Nothing can be done re: correlations.
            continue

        # Find ratings for the basis movies and the other movie
        joint_ratings_basis = []
        for bi in basis:
            joint_ratings_basis.append(
                [movie_vecs[bi][v] for v in joint_viewers])
        joint_ratings_other = [movie_vecs[oi][v] for v in joint_viewers]
        
        corr, pv = multi_corr(joint_ratings_basis, joint_ratings_other)
        if np.isnan(corr):
            # No point continuing, partitioning into too small sets.
            break
        if basis == [2152] and oi == 788:
            bug()
        if pv > 0.05:
            # No decent correlation. Add to our basis.
            print 'Adding', oi, movie_names[oi], 'to basis', pv
            basis.append(oi)
            basis_audience = reduce(
                set.intersection,
                (viewer_set[bi] for bi in basis))

            # Obviously the movie is contained within its island
            islands[oi].add(oi)

        else:
            # There's a correlation between ALL the movies in the basis set and
            # this movie. It would be nice if it was very similar to ONE of the
            # basis movies, coz then we can fatten up that basis movie with this
            # one. So check that condition.
            # for bi, ratings in zip(basis, joint_ratings_basis):
            #     dist = (len(viewer_set[bi].intersection(viewer_set[oi])) /
            #          float(len(viewer_set[bi].union(viewer_set[oi]))))
            #     corr, pv = stats.spearmanr(ratings, joint_ratings_other)
            #     if dist > 0.2 and pv < 0.05:
            #         islands[bi].add(oi)
            #     elif dist > 0.1 and pv <= 0.05:
            
            #     elif dist <= 0.1 and pv < 0.05:
            pass

    # Generating islands.
    print 'starting islands...'
    for o in xrange(len(movies)):
        oi = movies[o]
        a = 0
        for bi in basis:
            who_saw_both = viewer_set[bi].intersection(viewer_set[oi])
            who_saw_either = viewer_set[bi].union(viewer_set[oi])
            
            sim = len(who_saw_both) / float(len(who_saw_either))

            if sim > 0.1:
                # # Find joint ratings
                joint_bi = [movie_vecs[bi][v] for v in who_saw_both]
                joint_oi = [movie_vecs[oi][v] for v in who_saw_both]
                corr, pv = stats.spearmanr(joint_bi, joint_oi)

                # # If audience is similar *and* ratings correlated, same island
                if pv < 0.05:
                    # print movie_names[bi], movie_names[oi], corr, pv
                    islands[bi].add(oi)
                    a += 1
                else:
                    print 'weird', movie_names[bi], movie_names[oi], corr, pv

            if a > 1:
                print "WARNING", movies[oi]
                bug()

    print 'done island generation'
    fat_basis_audience = set()
    for bi in basis:
        island_viewership = set()
        for mi in islands[bi]:
            island_viewership.update([mi])

        if not fat_basis_audience:
            fat_basis_audience = island_viewership
        else:
            fat_basis_audience.intersection_update(island_viewership)

        p.add_row([
            movie_names[bi],
            len(movie_vecs[bi]),
            len(islands[bi]),
            len(island_viewership)
        ])
    
    print p
    print 'total movies:', len(MOVIE_IDS)
    print 'size of basis audience:', len(basis_audience)
    print 'size of island audience', len(fat_basis_audience)

    for bi in basis:
        print 'BASIS', movie_names[bi], '-'*100
        for i in islands[bi]:
            print movie_names[i]
'''

rvs = None
def lsh_sketch(list_vec):
    d = len(list_vec)
    vec = np.array(list_vec)
    global rvs
    if rvs is None:
        rvs = np.random.rand(FAMILY_SIZE, d)-0.5
    sketch = np.sign(rvs.dot(vec))
    return sketch


def signature_of(vec):
    sketch = lsh_sketch(vec)
    num = 0
    counter = 0
    for k in sketch:
        if k >= 0:
            num |= (1 << counter)
        counter += 1
    return num

# def num_bits_set(num):
#     # print 'wtf', num, np.log2(num)
#     x = int(2**np.ceil(np.log2(num)))
#     num_set = 0
#     while x > 0:
#         num_set += (num & x) > 0
#         x >>= 1
#     return num_set
# assert num_bits_set(16) == 1
# assert num_bits_set(17) == 2
# assert num_bits_set(31) == 5

def nxor(a, b):
    x = 1<<(FAMILY_SIZE-1)
    num_set = 0
    for i in xrange(FAMILY_SIZE):
        num_set += ((a&x)>0) != ((b&x)>0)
        x >>= 1
    # print num_set
    return num_set

assert nxor(0b1011, 0b1010) == 1
assert nxor(0b10, 0b1) == 2
assert nxor(0b10101010, 0b01010101) == 8


def multi_corr(xs, y):
    """Is there a pattern between likes/dislikes in xs, and likes/dislikes of y.

    Args:
        xs: list of lists of user ratings.
            [
                [m1_u1_rating, m1_u2_rating, m1_u3_rating, ...],
                [m2_u1_rating, m2_u2_rating, m2_u3_rating, ...],
                ...
            ]
        y: list of user ratings
            [u1_rating, u2_rating, u3_rating, ...]

    """

    # Assert same number of users for all ratings lists
    assert len(set(map(len, xs))) == 1
    assert len(xs[0]) == len(y)

    num_users = len(xs[0])
    num_x_movies = len(xs)

    # 2^num_movies buckets, for each combination of like/dislike
    buckets = [[] for i in range(2**len(xs))]
    for u in xrange(num_users):
        bucket_combo = [int(xs[m][u] >= 3) for m in xrange(num_x_movies)]
        bucket_signature = int(''.join(map(str, bucket_combo)), 2)
        buckets[bucket_signature].append(y[u])

    return stats.f_oneway(*buckets)


def cached(f):
    name = 'cache/{}_{}-{}_{}_cache.pkl'.format(f.__name__, START_MID,
                                                END_MID, TRAINING_SET)
    
    def writeToCache(data):
        with open(name, 'wb') as fout:
            cPickle.dump(data, fout)
        return data

    def helper(*args):
        assert not args  # This doesn't cache funcs with arguments
        
        if not os.path.exists(name):
            return writeToCache(f())
        else:
            try:
                print 'reading', f.__name__, 'from cache'
                return cPickle.load(open(name, 'rb'))
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
    # us = set()
    for i, fn in movie_filenames:
        f = open(TRAINING_SET + '/' + fn)
        # fo = open('training_med/' + fn, 'w')
        # fo.write(f.readline())  # Ignore first line, which contains "id:"
        f.readline()
        d = {}
        
        for line in f:
            # if not line: break
            # print line
            uid, rating, date = line.split(',')
            # if int(uid) < 100000:
            #     fo.write(line)
            #     us.add(uid)
            d[int(uid)] = int(rating)
        f.close()
        # fo.close()
        # print i, len(us)
        movie_vectors[i] = d
    print 'loaded ratings from training data'
    return movie_vectors

@cached
def get_movie_names():
    """A list whose ith element is the name of movie i."""
    movie_names = [-1337]
    for line in open('movie_titles.txt'):
        i, year, name = line.strip().split(',', 2)
        movie_names.append(name)
    return movie_names




# def intersect_size(m1, m2):
#     return len(set(m1).intersection(m2))


# def correlation(m1, m2):
#     """
#     Find how many users saw both a and b.
#     a, b are dicts of userid : rating.
#     returns (correlation strength 0 to 1, p-value 0 to 1).

#     """
#     users_who_saw_both = set(m1).intersection(m2)
#     if len(users_who_saw_both) < 3:
#         # not enough data points for correlation
#         return (None, None)
#     x = [m1[u] for u in users_who_saw_both]
#     y = [m2[u] for u in users_who_saw_both]

#     return stats.spearmanr(x, y)

complete()
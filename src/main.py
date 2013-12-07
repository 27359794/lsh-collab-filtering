"""
main.py
Daniel Goldbach
2013 October

Run collaborative filtering on Netflix challenge dataset and output RMSE.

For quicker debugging time, uncomment the @cached decorations above functions
that can be cached.
"""

import argparse
import numpy as np
import os
import scipy.sparse
import collections

# Project-specific imports
from cosine_nn import CosineNN
import utils
import config


"""
A class representing an entry in the inverse index {uid: user data}. `mean' and
`std' are only included for debugging (when you need to know a user's movie
rating from their normalised movie rating. They aren't used by the algorithm at
the moment).
"""
IIndexEntry = collections.namedtuple('IIndexEntry', ['mean', 'std', 'uratings'])


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


def index_and_evaluate(movie_ids):
    """Run the collaborative filtering algorithm on the movies in movie_ids."""
    iindex = get_normalised_inverse_index(movie_ids)
    user_ids = iindex.keys()
    nn_index = CosineNN(max(movie_ids) + 1) # +1 since ids are 1-based

    # Add all movies to nn_index
    for i, uid in enumerate(user_ids):
        col = iindex[uid].uratings
        nn_index.index(uid, col)
        if i % 10 == 0:
            print('index progress: {:.3%}'.format(i / len(user_ids)))
    print('indexing and setup complete')

    # Find all test data instances for which we can predict a rating
    probe_ratings = read_probe(movie_ids)
    to_predict = []
    for mid, mratings in probe_ratings.items():
        for uid, rating in mratings.items():
            if uid in user_ids:
                to_predict.append((uid, mid, rating))

    # Predict ratings for test data instances and add up error
    errors = []
    for i, (uid, mid, actual) in enumerate(to_predict):
        print('evaluation progress: {:.3%}'.format(i / len(to_predict)))
        guess = guess_rating(nn_index, iindex, uid, mid)
        errors.append((guess - actual)**2)
    print('RMSE:', np.sqrt(np.mean(errors)))


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


@utils.cached
def read_probe(movie_ids):
    """A list whose ith element is a dict {user: rating} for movie i."""
    probe_file = open(os.path.join(config.RATED_PROBE_FN))
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


@utils.cached
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
    user_index = collections.defaultdict(list)
    for mid in sorted_movie_ids:
        for (uid, rating) in movie_ratings[mid].items():
            user_index[uid].append((mid, rating))

    for uid, udata in user_index.items():
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
        sparse_ratings = scipy.sparse.csc_matrix(user_vec).T
        iindex[uid] = IIndexEntry(mean=umean, std=ustddev,
                                  uratings=sparse_ratings)

    return iindex


@utils.cached
def get_movie_ratings(movie_ids):
    """@return {movieid : {userid: rating}"""
    movie_filenames = [(mi, 'mv_{:0>7}.txt'.format(mi)) for mi in movie_ids]
    movie_vectors = {}
    for i, filen in movie_filenames:
        assert os.path.exists(os.path.join(config.TRAINING_SET_DIR, filen)), \
               'ensure dataset containing {} has been generated'.format(filen)
        movie_file = open(os.path.join(config.TRAINING_SET_DIR, filen))
        movie_file.readline()

        uid_to_rating = {}
        for line in movie_file:
            uid, rating, date = line.split(',')
            uid_to_rating[int(uid)] = int(rating)
        movie_vectors[i] = uid_to_rating
        movie_file.close()
    print('loaded ratings from training data')
    return movie_vectors


if __name__ == '__main__':
    main()

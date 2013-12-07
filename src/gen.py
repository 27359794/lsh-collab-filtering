"""
gen.py
Daniel Goldbach
2013 October

Generate properly formatted datasets.
"""

import argparse
import collections
import os

import config


def main():
    parser = argparse.ArgumentParser(
        description='Generate formatted and appropriately sized training sets.')
    parser.add_argument(
        'num_movies',
        type=int,
        help='the number of movies to include in the datasets. '
             '1 <= num_movies <= 17770.')
    parser.add_argument(
        '-f', '--fulldataset',
        action='store_true',
        help='use the full dataset as opposed to the medium dataset.')
    args = parser.parse_args()

    training_set_fn = (config.RAW_TRAINING_LRG_DIR if args.fulldataset
                       else config.RAW_TRAINING_MED_DIR)
    movie_ids = range(1, args.num_movies + 1)

    # Create necessary directories if they don't exist
    for directory in config.GENERATED_DIRS:
        if not os.path.exists(directory):
            os.makedirs(directory)

    generate(movie_ids, training_set_fn)

    print('successfully generated datasets.')


def generate(mids, training_set):
    """
    Generates disjoint and nicely formatted training and test sets from the
    Netflix challenge data. The test set contains data from Netflix's probe set.
    The original challenge data combines all ratings into the training set,
    which doesn't make sense to me. Better to have a clear separation between
    training and test data.

    @param mids
        list of movie IDs to include in generated sets
    @param training_set
        str path to directory containing training data per movie
    """

    probe_file = open(config.RAW_PROBE_FN)
    probe = collections.defaultdict(list)
    mid = None
    for line in probe_file:
        if ':' in line:
            mid = int(line.split(':')[0])
        else:
            uid = int(line)
            probe[mid].append(uid)

    formatted_probe_file = open(config.RATED_PROBE_FN, 'w')

    for mid in mids:
        name = 'mv_{:0>7}.txt'.format(mid)

        if mid in probe:
            formatted_probe_file.write('%d:\n' % mid)

        training_in = open(os.path.join(training_set, name))
        training_out = open(os.path.join(config.TRAINING_SET_DIR, name), 'w')

        # First line always contains movie ID then ':'
        training_out.write(training_in.readline())
        for line in training_in:
            uid = int(line.split(',')[0])

            # Write instance into either training set or test set based on type
            if uid in probe[mid]:
                formatted_probe_file.write(line)
            else:
                training_out.write(line)


if __name__ == '__main__':
    main()

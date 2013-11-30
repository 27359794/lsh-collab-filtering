"""
cosine_nn.py
Daniel Goldbach
2013 October

Nearest neighbours data structure using cosine distance metric.
"""

import numpy as np
from collections import defaultdict
import scipy.spatial
import doctest

import config


class CosineNN(object):
    """
    Nearest neighbours data structure using cosine distance metric.

    vector m is a neighbour of vector n if the angle between m and n is less
    than a.

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
    (pi/3, pi/2,( 1-1/3)^4, (1-1/2)^4) = (pi/3, pi/2, .20, 0.06)
    Applying a union over 6 blocks, we get
    (pi/3, pi/2, 1-(1-(1-1/3)^4)^6, 1-(1-(1-1/2)^4)^6) = (.., .73, 0.32)

    Using block size 9 with 65 blocks, we get good values
    In [31]: 1-(1-(1-1/3.0)**9)**65, 1-(1-(1-1/2.0)**9)**65
    Out[31]: (0.819708249125523, 0.11933437406869052)
    i.e. a (pi/3, pi/2, 0.82, 0.12)-sensitive hash family.
    There's a >=0.82 chance of neighbours being in the result set,
    and a <=0.12 chance of non-neighbours being in the result set.

    OR gate of AND gates:
    a = k
    o = l

    p = probability that 1 pair of bits match = 1 - angle/pi

    p(block of length a matches all)    = p^a
    p(block of length o matches once)   = 1-(doesn't match all)
                                        = 1 - P(doesn't match once)^o
                                        = 1 - (1-p)^o
    OR gate of AND gates:
        1 - (1 - (1-1/3.0)**a)**o, 1 - (1 - (1 - 1/2.0)**a)**o
    a,o = 7,60
    Out[78]: (0.9731803062239186, 0.37536677884638525)
    """

    BLOCK_SIZE, NUM_BLOCKS = config.BLOCK_SIZE, config.NUM_BLOCKS
    SIG_LENGTH = BLOCK_SIZE * NUM_BLOCKS

    def __init__(self, vector_len):
        self.col_vecs = {}
        self.signatures = {}

        # CosineNN.SIG_LENGTH * vector_len matrix where each column is a vector
        # corresponding to a random hyperplane in the family. All entries are in
        # [-0.5, 0.5).
        self.random_vector_family = np.matrix(
            np.random.rand(CosineNN.SIG_LENGTH, vector_len) - 0.5)

        # nn_index[(which_block, block_integer_value)] = list of m-ids in bucket
        self.nn_index = defaultdict(list)

    def index(self, iid, col):
        """
        Indexes item with ID `iid', column vector `col' into data structure.
        """
        # Form the column vector of the utility matrix for m
        sig = self.signature_of(col)

        self.signatures[iid] = sig
        self.col_vecs[iid] = col

        # Index this signature
        for block_num in xrange(CosineNN.NUM_BLOCKS):
            block_val = extract_block(sig, block_num)
            self.nn_index[(block_num, block_val)].append(iid)

    def query(self, iid):
        """
        Get approximate nearest neighbours for item with id iid.

        Time complexity of this function is slightly weird. It's something like
        O(p_2*M + p_1*result) with plenty of overhead, where p_1, p_2 are the
        probability values of the (d_1, d_2, p_1, p_2) hash family in use.
        """
        sig = self.signatures[iid]
        resultset = set()
        for block_num in xrange(CosineNN.NUM_BLOCKS):
            block_val = extract_block(sig, block_num)
            resultset.update(self.nn_index[(block_num, block_val)])
        return resultset


    def query_with_dist(self, iid):
        """
        Returns a set of (id,cosdist) to potential neighbours of object with id
        `iid'. Does NOT return iid as one of the results.

        Slower than query() because it computes all the cosines.
        """
        maybe_neighbours = self.query(iid)
        with_dist = [(niid, self.cosine_dist_between(iid, niid))
                     for niid in maybe_neighbours if niid != iid]
        return with_dist

    def signature_of(self, vec):
        """Takes a numpy vector of length U and produces its LSH."""
        sketch = self.random_vector_family * vec  # This is a matrix product
        num = 0
        # Generate the signature (an integer)
        # TODO: find a way of vectorising this loop.
        for i in xrange(CosineNN.SIG_LENGTH):
            if sketch[i, 0] >= 0:
                num |= (1 << i)
        return num

    def cosine_dist_between(self, iid1, iid2):
        """
        Calculates exact cosine DISTANCE i.e. 1 - cosine similarity between
        items with IDs iid1, iid2.
        """
        # Requires .todense() or else 'dimension missmatch'
        return scipy.spatial.distance.cosine(self.col_vecs[iid1].todense(),
                                             self.col_vecs[iid2].todense())


def extract_block(sig, block_num):
    """
    Extract the block_numth block of binary bits (starting from the left) from
    the integer sig and return result as int.

    Shift the sig right until the last BLOCK_SIZE bits are the block, then take
    off the bits before the block.

    >>> CosineNN.BLOCK_SIZE = 3
    >>> extract_block(0b111010110, 0) ==  0b110
    True
    >>> extract_block(0b111010110, 1) == 0b010
    True
    >>> extract_block(0b111010110, 2) == 0b111
    True
    """
    return (sig >> (block_num*CosineNN.BLOCK_SIZE)) % (1 << CosineNN.BLOCK_SIZE)


if __name__ == '__main__':
    doctest.testmod()

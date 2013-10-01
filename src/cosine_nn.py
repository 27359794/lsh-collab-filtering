"""
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
(pi/3, pi/2,( 1-1/3)^4, (1-1/2)^4) = (pi/3, pi/2, .20, 0.06)
Applying a union over 6 blocks, we get
(pi/3, pi/2, 1-(1-(1-1/3)^4)^6, 1-(1-(1-1/2)^4)^6) = (.., .73, 0.32)

Using block size 9 with 65 blocks, we get good values
In [31]: 1-(1-(1-1/3.0)**9)**65, 1-(1-(1-1/2.0)**9)**65
Out[31]: (0.819708249125523, 0.11933437406869052)
i.e. a (pi/3, pi/2, 0.82, 0.12)-sensitive hash family.
There's a >=0.82 chance of neighbours being in the result set,
and a <=0.12 chance of non-neighbours being in the result set.
"""

import numpy as np
from collections import defaultdict
import scipy.spatial


class CosineNN(object):
    BLOCK_SIZE = 13#12
    NUM_BLOCKS = 300#70
    SIG_LENGTH = BLOCK_SIZE * NUM_BLOCKS

    def __init__(self, vector_len):
        self.col_vecs = {}
        self.signatures = {}
        self.random_vector_family = np.random.rand(CosineNN.SIG_LENGTH, vector_len) - 0.5

        # nn_index[(which_block, block_integer_value)] = list of m-ids in bucket
        self.nn_index = defaultdict(list)

    def index(self, mid, col):
        """mid is the identifier for the column. col is a column vector."""
        # Form the column vector of the utility matrix for m
        sig = self.signature_of(col)

        self.signatures[mid] = sig
        self.col_vecs[mid] = col

        # Index this signature
        for block_num in xrange(CosineNN.NUM_BLOCKS):
            block_val = self.extract_block(sig, block_num)
            self.nn_index[(block_num, block_val)].append(mid)

    def query(self, mid):
        """
        Time complexity of this function is slightly weird. It's something
        like O(p_2*M + p_1*result) with plenty of overhead, where p_1, p_2 are
        the probability values of the (d_1, d_2, p_1, p_2) hash family in use.

        """
        sig = self.signatures[mid]
        resultset = set()
        for block_num in xrange(CosineNN.NUM_BLOCKS):
            block_val = self.extract_block(sig, block_num)
            resultset.update(self.nn_index[(block_num, block_val)])
        return resultset


    def find_neighbours(self, mid, eps):
        """
        This is an exact version of query(). Precision guaranteed 100%.
        Recall is the same as the recall of query(). Also significantly slower
        because it computes all the cosines.

        """
        maybe_neighbours = self.query(mid)
        actual_neighbours = {nmid for nmid in maybe_neighbours
                             if self.cosine_between(mid, nmid) < eps}
        return actual_neighbours


    def extract_block(self, sig, block_num):
        # Shift the sig right until the last BLOCK_SIZE bits are the block,
        # Then take off the bits before the block.
        return (sig >> (block_num*CosineNN.BLOCK_SIZE)) % (1 << CosineNN.BLOCK_SIZE)


    def signature_of(self, vec):
        """Takes a numpy vector of length U and produces its LSH."""
        sketch = self.random_vector_family.dot(vec)
        num = 0
        counter = 0
        for k in sketch:
            if k >= 0:
                num |= (1 << counter)
            counter += 1
        return num

    def nxor_count_bits(self, a, b):
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
        for _ in xrange(SIG_LENGTH):
            num_set += ((a&x)>0) != ((b&x)>0)
            x <<= 1
        return num_set

    def cosine_between(self, mid1, mid2):
        return scipy.spatial.distance.cosine(self.col_vecs[mid1], self.col_vecs[mid2])

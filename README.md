Efficient Collaborative Filtering with LSH
==========================================

* Quick link to [the project report](http://www.cse.unsw.edu.au/~dgol478/report.pdf).

Requirements
------------

* Python 2.7.x
* numpy + scipy modules
* Download and extract [`data.zip`](https://dl.dropboxusercontent.com/u/1103246/data.zip) (720MB) or [`data.7z`](https://dl.dropboxusercontent.com/u/1103246/data.7z) (460MB) into the root directory of the project


Instructions
------------

1. Unzip the contents of `data.zip` into the root directory. There should now be files `movie_titles.txt`, `probe.txt` and folders `training_set`, `training_med` in the root directory.
2. `cd` to `src/`
3. Run `python gen.py` with appropriate command line arguments to generate disjoint training and test sets. `python gen.py -h` gives further information.
4. Run `python code.py`. This will create the nearest neighbour data structure using the training set you generated in the previous command, then it will attempt to predict ratings for test set users and calculate the error.
5. The last line printed by the script is the RMSE on the probe set.

The *k,l* parameters described in the report can be configured in `src/cosine_nn.py`.
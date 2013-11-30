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

1. Unzip the contents of compressed file `data.(zip|7z)` into the root directory. There should now be files `movie_titles.txt`, `probe.txt` and folders `training_set`, `training_med` in the root directory.
2. `cd` to `src/`
3. Run `python gen.py` with appropriate command line arguments to generate disjoint training and test sets. `python gen.py -h` gives further information.
4. Run `python main.py` with appropriate command line arguments (`-h` flag for help). This will create the nearest neighbour data structure using the training set you generated in the previous command, then it will attempt to predict ratings for test set users and calculate the error.
5. The last line printed by the script is the RMSE on the probe set.

The *k*,*l* parameters described in the report can be configured as constants in `src/cosine_nn.py`.

Sample Usage
------------

```
antares: src\ $ python gen.py 300
successfully generated datasets.
antares: src\ $ python main.py 300
index progress: 0.000%
index progress: 0.917%
index progress: 1.835%
...
index progress: 97.248%
index progress: 98.165%
index progress: 99.083%
indexing and setup complete
evaluation progress: 0.000%
evaluation progress: 1.010%
evaluation progress: 2.020%
...
evaluation progress: 96.970%
evaluation progress: 97.980%
evaluation progress: 98.990%
RMSE: 1.05259425337
```

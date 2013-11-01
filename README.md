Requirements
------------

* Python 2.7.x
* numpy + scipy modules


Instructions
------------

1. Unzip the contents of `data.zip` into the root directory. There should now be a file `probe_rated.txt` and a folder `noprobe` in the root directory.
2. Run `python src/code.py`. By default this will run CF on the first 1000 movies of the *small* training set. To switch to the full training set, change `noprobe/training_med` in `code.py` to `noprobe/training_set`. To process a different set of movies, change `START_MID` and `END_MID` (first and last movie ID respectively) in code.py.
3. The last line printed by the script is the RMSE on the probe set.

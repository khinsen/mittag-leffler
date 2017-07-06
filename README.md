# The Mittag-Leffler function in Python

This is a Python port of
a published
[Matlab implementation](https://se.mathworks.com/matlabcentral/fileexchange/48154-the-mittag-leffler-function) of
the generalized Mittag-Leffler function.

The script `test_ml.py` contains pretty much all the tests that I have
run until now. They cover very little of the total functionality of
the code. My personal needs are limited to `beta=1`, `gamma=1`, and
`z` real, so I might never get around to add decent tests for
everything. Contributions are welcome.

The module `ml_internal.py` contains the parts that I would like to
compile using [Pythran](https://github.com/serge-sans-paille/pythran)
for speed, but I haven't succeeded yet. Client code should only import
module `mittag_leffler`.

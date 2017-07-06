import numpy as np
from scipy.special import erfc
from mittag_leffler import ml

z = np.linspace(-2., 2., 50)
assert np.allclose(ml(z, 1.), np.exp(z))

z = np.linspace(-2., 2., 50)
assert np.allclose(ml(z**2, 2.), np.cosh(z))

z = np.linspace(0., 2., 50)
assert np.allclose(ml(np.sqrt(z), 0.5), np.exp(z)*erfc(-np.sqrt(z)))

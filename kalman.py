
"""
Python module to implement Kalman filters for multiple data streams.

Suppose you have n sensors each out-putting a value every time step.

One simple way to 'smooth out' noise and rapid perturbations in a signal is to
implement an exponentially-weighted average (EWA) also known as an
exponentially-weighted moving average (EWMA). At each timestep you calculate
an EWA value for each parameter based on a weigthed sum of the previous EWA
the current value of x as follows:

values = beta*value_prev + (1 - beta)*x

where beta is a parameter that determines how much to weight previous values
compared to the latest value (e.g. beta = 0.9).

For more info see https://en.wikipedia.org/wiki/Exponential_smoothing.

You might want to have a different beta value for each sensor input so beta is
an array of floats of length n.

See main() function below for a demonstration of how to use this class.
"""

import numpy as np
from math import log


class EWAFilter:

    def __init__(self, n, beta=0.9, dtype=np.float, bias_correction=True):
        """
        Exponentially-weighted average filter for multiple data streams.

        Args:
            n (int): Number of parallel data streams
            beta (float): The weighting coefficient (between 0 and 1).
                          Determines how much to weight previous values
                          compared to the latest value (a higher beta
                          discounts older observations faster).
            dtype (numpy.dtype): Data type (should be a floating-point
                                 type)
            bias_correction (bool): Turn bias correction on or off
                                    (True by default)
        """

        self.n = n
        self.beta = beta
        self.dtype = dtype
        self.values = np.zeros(n, dtype=dtype)  # v[0] = 0
        self.t = 0
        self._uvalues = None
        self.bias_correction = bias_correction
        if bias_correction:
            self._uvalues = np.zeros(n, dtype=dtype)
            # Put a limit on how long bias correction is used
            # Once beta**t < 0.5e-6 its effect is zero
            min_beta = max(beta, 0.9)
            self.beta = max(self.beta, min_beta)
            self._correction_duration = int(log(0.5e-6)/log(min_beta))

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):

        if isinstance(value, float):
            self._beta = value
        else:
            try:
                if len(value) == self.n:
                    self._beta = np.array(value, dtype=np.float)
                else:
                    raise ValueError
            except:
                raise ValueError("beta must be a float or sequence of "
                                 "floats of length n")

    def process(self, x):
        """Process a new set of data points and update averages.

        Args:
            x (np.ndarray): One-dimensional array or sequence of new
                            data values (length self.n).
        """

        assert isinstance(x, np.ndarray), "x must be an array"
        assert len(x) == self.n, "x must be length n"

        # First data arrives at t=1
        self.t += 1

        if not self.bias_correction or self.t > self._correction_duration:

            # v[t] = beta*v[t-1] + (1 - beta)*x[t]
            self.values = self.beta*self.values + \
                          (1 - self.beta)*np.array(x, dtype=self.dtype)

        else:
            # Keep a copy of 'uncorrected' values in _uvalues
            self._uvalues = self.beta*self._uvalues + \
                            (1 - self.beta)*np.array(x, dtype=self.dtype)

            # v_corrected[t] = v[t]/(1 - beta**t)
            self.values = self._uvalues/(1 - self.beta**self.t)

    def reset(self):
        """Reset the filter to its initialized state (t = 0)."""

        self.t = 0
        self.values[:] = 0.0
        if self.bias_correction:
            self._uvalues[:] = 0.0


def main():
    """Demonstrate use of EWA Filter."""

    import time

    # Size of input array
    n = 5

    # Initialize EWA filter
    ewa_filter = EWAFilter(n, beta=0.8)

    # Begin filtering data
    for i in range(25):

        # Simulate noisy data
        x = np.random.randn(n)
        ewa_filter.process(x)

        print(ewa_filter.t, ewa_filter.values)
        time.sleep(0.5)


if __name__ == "__main__":
    main()

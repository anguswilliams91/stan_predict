"""A simple example of how one might use samples from Stan for prediction.

This example uses a simple model

y ~ N(theta_0 + theta_1 * X, 1).

A stan model is fitted, and the resulting posterior samples are cached.
Then, when predictions are required, a separate piece of stan code with only a
generated quantities block is called, and the posterior samples of the model
parameters are passed in the data block, alongside the new data.

One interesting issue that I ran into here was that if I tried to simulate
posterior predictive samples for multiple datapoints, the method

fit.extract()

in pystan was *very slow*. I had to write stan code that produced samples
for a single datapoint, and then do a python list comprehension over all of
the datapoints in order to avoid this issue. I'm not exactly sure why this is,
but it is prohibitive and I should maybe look into it.
"""
import os

import numpy as np
import pystan


def simulate_data(theta_0, theta_1, n=100, seed=42):
    """Simulate some data."""
    np.random.seed(seed)
    X = np.random.uniform(low=-10, high=10, size=n)
    mu = theta_0 + theta_1 * X
    y = np.array([np.random.normal(loc=mu[i]) for i, _ in enumerate(X)])
    return X, y


class StanPredictor:
    """Simple helper class for fitting a Stan model then predicting later.

    The model is a 1D linear regression with unit normal uncertainty.
    """

    def __init__(self, seed=42):
        """Set up model."""
        self.seed = seed
        self.theta_0 = None
        self.theta_1 = None
        self.model = None

    def fit(self, X, y, thin=10, return_fit=False):
        """Fit to the training data."""
        stan_data = dict(
            X=X,
            y=y,
            n=len(y)
        )
        self.X_train = X
        self.y_train = y
        print("Compiling and fitting stan model...")
        with suppress_stdout_stderr():
            model = pystan.StanModel(file="fit.stan")
            fit = model.sampling(data=stan_data, seed=self.seed)
        self.theta_0 = fit["theta_0"].ravel()[::thin]
        self.theta_1 = fit["theta_1"].ravel()[::thin]
        if return_fit:
            return fit
        else:
            return None

    def _predict_single(self, X):
        """Produce samples from the predictive distribution.

        Draw samples from p(y | X, X_train, y_train).
        """
        stan_data = dict(
            X=X,
            n=len(self.theta_0),
            theta_0=self.theta_0,
            theta_1=self.theta_1
        )
        if self.model is None:
            self.model = pystan.StanModel(file="predict.stan")
        fit = self.model.sampling(
            data=stan_data,
            seed=self.seed,
            iter=1,
            chains=1,
            algorithm="Fixed_param"
        )
        return fit["y_pred"].ravel()

    def predict(self, X):
        """Predict for new observations.

        Note this will run slower the first time it is used, because the
        underlying stan code must be compiled.
        """
        with suppress_stdout_stderr():
            # suppress messages from stan as the predictions are made.
            samples = np.array([self._predict_single(Xi) for Xi in X])
        return samples


class suppress_stdout_stderr(object):
    """A context manager for doing a "deep suppression" of stdout and stderr.

    Taken from https://github.com/facebook/prophet/issues/223.
    """

    def __init__(self):
        """Open a pair of null files."""
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        """Assign the null pointers to stdout and stderr."""
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        """Re-assign the real stdout/stderr back to (1) and (2)."""
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

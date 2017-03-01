from multiprocessing import Pool
from itertools import repeat

from numpy.random import randint
from pcanet import PCANet
import numpy as np
from tqdm import tqdm



def transform(estimator, images):
    return estimator.transform(images)


def fit_random(estimator, images, sampling_ratio):
    n_images = images.shape[0]
    n_samples = int(n_images * sampling_ratio)
    indices = randint(0, n_images, n_samples)
    estimator.fit(images[indices])
    return estimator


class Bagging(object):
    def __init__(self, n_estimators, sampling_ratio, n_jobs,
                 **estimator_params):
        """
        n_estimators: int
        sampling_ratio: int
            The number of samples to draw from X to train each base estimator.
        n_jobs: int
            The number of jobs to run in parallel.
            The number of cores is set if -1.
        estimator_params: dict
            Parameters for PCANet.__init__
        """

        self.estimators = \
            [PCANet(**estimator_params) for i in range(n_estimators)]
        # Validate only the first estimator
        # since all of estimators share the same hyperparameters
        self.estimators[0].validate_structure()

        self.sampling_ratio = sampling_ratio
        self.n_jobs = n_jobs

    def fit(self, images):
        # run fit_random(estimator, images, sampling_ratio) in parallel
        args = zip(self.estimators,
                   repeat(images),
                   repeat(self.sampling_ratio))

        with Pool(self.n_jobs) as pool:
            self.estimators = pool.starmap(fit_random, args)
        return self.estimators

    def transform(self, images):
        # run transform(estimator, images) in parallel
        with Pool(self.n_jobs) as pool:
            X = pool.starmap(transform, zip(self.estimators, repeat(images)))
        return np.mean(X, axis=0)  # merge the results

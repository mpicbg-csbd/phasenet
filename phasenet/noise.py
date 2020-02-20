import numpy as np
import matplotlib.pyplot as plt
from csbdeep.utils import _raise

class NoiseGenerator:

    def __init__(self, image):

        self.img = image
        self.img = (self.img - np.min(self.img)) / (np.max(self.img) + np.min(self.img))

    def _normal_noise(self, mean, sigma, snr):
        noisy = np.random.normal(mean, sigma, self.img.shape) + self.img * mean * snr
        return noisy

    def _poisson_noise(self, snr):

        lambdaDistribution = self.img  * snr
        noisy = np.random.poisson(np.maximum(1, lambdaDistribution + 1).astype(int)).astype(np.float32)
        return noisy

    def add_normal_poisson_noise(self, mean, sigma, snr):

        normal_noise_img = self._normal_noise(mean=mean, sigma=sigma, snr=snr)
        poisson_noise_img = self._poisson_noise(snr=snr)
        _noisy_img = normal_noise_img + poisson_noise_img
        return _noisy_img


def add_random_noise(image, params, rng=None):
    """
        add gaussian and poisson noise to the system
        :param image: 3D array as image
        :param params: dictionary expecting scalar or list/tuples for snr, mean, sigma
        :return:

    """
    if rng is None: rng = np.random

    n = NoiseGenerator(image)
    all((np.isscalar(v) and v>=1) or (isinstance(v,(tuple,list)) and len(v)==2) for v in params.values()) or _raise(ValueError())
    params = {k:((v,v) if np.isscalar(v) else v) for k,v in params.items()}
    all(v[0]<=v[1] for v in params.values()) or _raise(ValueError("Lower bound is expected to be less than the upper bound"))
    mean = np.random.uniform(*params['mean']) if 'mean' in params else _raise(ValueError("No value for mean"))
    sigma = np.random.uniform(*params['sigma']) if 'sigma' in params else _raise(ValueError("No vlaue for sigma"))
    snr = np.random.uniform(*params['snr']) if 'snr' in params else _raise(ValueError("No value for SNR"))

    noisy = n.add_normal_poisson_noise(mean=mean, sigma=sigma, snr=snr)
    noisy = np.maximum(0, noisy)

    return noisy
    




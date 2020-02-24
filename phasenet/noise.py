import numpy as np
import matplotlib.pyplot as plt
from csbdeep.utils import _raise

def normal_noise(image, mean, sigma, snr):
    noisy = np.random.normal(mean, sigma, image.shape) + image * mean * snr
    return noisy

def poisson_noise(image, snr):

    lambdaDistribution = image  * snr
    noisy = np.random.poisson(np.maximum(1, lambdaDistribution + 1).astype(int)).astype(np.float32)
    return noisy

def add_normal_poisson_noise(image, mean, sigma, snr):

    normal_noise_img = normal_noise(image=image, mean=mean, sigma=sigma, snr=snr)
    poisson_noise_img = poisson_noise(image=image, snr=snr)
    noisy_img = normal_noise_img + poisson_noise_img
    return noisy_img

# as discussed, make this return a function that can repeatedly been applied to an image to make it noisy
def add_random_noise(image, snr, mean, sigma, rng=None):
    """
        add gaussian and poisson noise to the system
        :param image: 3D array as image
        :param snr: scalar or tuple, signal to noise ratio
        :param mean: scalar or tuple, mean background noise
        :param sigma: scalar or tuple, simga for gaussian noise 
        :return: 3d array

    """
    if rng is None: rng = np.random

    snr = (snr, snr) if np.isscalar(snr) else snr
    mean = (mean, mean) if np.isscalar(mean) else mean
    sigma =(sigma, sigma) if np.isscalar(sigma) else sigma


    all(v[0]<=v[1] for v in [snr,mean,sigma]) or _raise(ValueError("Lower bound is expected to be less than the upper bound"))
    all(v[0]>=0 and v[1]>=0 for v in [snr,mean,sigma]) or _raise(ValueError("noise is expected to be greater than 0"))
    
    mean = np.random.uniform(*mean)
    sigma = np.random.uniform(*sigma) 
    snr = np.random.uniform(*snr)
    image = (image - np.min(image)) / (np.max(image) + np.min(image))
    noisy = add_normal_poisson_noise(image=image, mean=mean, sigma=sigma, snr=snr)
    noisy = np.maximum(0, noisy)

    return noisy
    




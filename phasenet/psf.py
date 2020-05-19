import numpy as np
from csbdeep.utils import _raise



class PsfGenerator3D:
    """
    3D PSF generator, courtesy of Martin Weigert (https://github.com/maweigert)
    """

    def __init__(self, psf_shape, units, lam_detection, n, na_detection, n_threads=4):

        """
        encapsulates 3D PSF generator

        :param psf_shape: tuple, psf shape as (z,y,x), e.g. (64,64,64)
        :param units: tuple, voxel size in microns, e.g. (0.1,0.1,0.1)
        :param lam_detection: scalar, wavelength in microns, e.g. 0.5
        :param n: scalar, refractive index, eg 1.33
        :param na_detection: scalar, numerical aperture of detection objective, eg 1.1
        :param n_threads: integer, for multiprocessing
        """

        psf_shape = tuple(psf_shape)
        units = tuple(units)

        # setting up the fourier domain and the wavefronts...

        self.Nz, self.Ny, self.Nx = psf_shape
        self.dz, self.dy, self.dx = units

        self.na_detection = na_detection
        self.lam_detection = lam_detection

        self.n = n

        kx = np.fft.fftfreq(self.Nx, self.dx)
        ky = np.fft.fftfreq(self.Ny, self.dy)

        z = self.dz * (np.arange(self.Nz) - self.Nz // 2)

        self.KZ3, self.KY3, self.KX3 = np.meshgrid(z, ky, kx, indexing="ij")
        KR3 = np.sqrt(self.KX3 ** 2 + self.KY3 ** 2)

        # the cutoff in fourier domain
        self.kcut = 1. * na_detection / self.lam_detection
        self.kmask3 = (KR3 <= self.kcut)

        H = np.sqrt(1. * self.n ** 2 - KR3 ** 2 * lam_detection ** 2)

        self._H = H

        out_ind = np.isnan(H)
        self.kprop = np.exp(-2.j * np.pi * self.KZ3 / lam_detection * H)
        self.kprop[out_ind] = 0.

        self.kbase = self.kmask3 * self.kprop

        KY2, KX2 = np.meshgrid(ky, kx, indexing="ij")
        KR2 = np.hypot(KX2, KY2)

        self.krho = KR2 / self.kcut
        self.kphi = np.arctan2(KY2, KX2)
        self.kmask2 = (KR2 <= self.kcut)

        self.myzifftn = lambda x: np.fft.ifftn(x,axes=(1,2))


    def masked_phase_array(self, phi, normed=True):
        """
        returns masked Zernike polynomial for back focal plane, masked according to the setup

        :param phi: Zernike/ZernikeWavefront object
        :param normed: boolean, multiplied by normalization factor, eg True
        :return: masked wavefront, 2d array
        """
        return self.kmask2 * phi.phase(self.krho, self.kphi, normed=normed, outside=None)


    def coherent_psf(self, phi, normed=True):
        """
        returns the coherent psf for a given wavefront phi

        :param phi: Zernike/ZernikeWavefront object
        :return: coherent psf, 3d array
        """
        phi = self.masked_phase_array(phi, normed=normed)
        ku = self.kbase * np.exp(2.j * np.pi * phi / self.lam_detection)
        res = self.myzifftn(ku)
        return np.fft.fftshift(res, axes=(0,))

    def incoherent_psf(self, phi, normed=True):
        """
        returns the incoherent psf for a given wavefront phi
           (which is just the squared absolute value of the coherent one)
           The psf is normalized such that the sum intensity on each plane equals one

        :param phi: Zernike/ZernikeWavefront object
        :return: incoherent psf, 3d array
        """
        _psf = np.abs(self.coherent_psf(phi, normed=normed)) ** 2
        _psf = np.array([p/np.sum(p) for p in _psf])
        return np.fft.fftshift(_psf)

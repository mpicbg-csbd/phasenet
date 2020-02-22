from csbdeep.utils import _raise
import numpy as np
import random
import warnings
from abc import ABC, abstractmethod


class Phantoms3D(ABC):

    def __init__(self, shape):
        self.shape=shape
        len(self.shape)==3 or _raise(ValueError("Only 3D phantoms are supported"))
        self.phantom_obj = np.zeros(self.shape)

    def check_sum_of_phantom_obj(self):
        if np.sum(self.phantom_obj) <= 0:
            warnings.warn("No object created")

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def get(self):
        pass


class Points(Phantoms3D):

    """
        creates multiple points
        :param obj_shape: tuple, object shape as (z,y,x), e.g. (64,64,64)
        :param num: integer, number of points, e.g. 3
        :param center: boolean, whether to have a point at the center, default is False
        :param pad_from_boundary: integer, leave space between points and bouandary. Helpful for convolution, recommended size//5
    """

    def __init__(self, shape, num, center = True, pad_from_boundary=0):
        super().__init__(shape)
        self.num = num
        self.center = center
        self.pad_from_boundary = pad_from_boundary
        self.generate()

    def generate(self):
        _num = self.num
        np.isscalar(self.pad_from_boundary) and self.pad_from_boundary<np.min(self.shape)//2  or _raise(ValueError( "Padding from boundary has to be scalar and bouded by object size"))
        
        x = np.zeros(self.shape, np.float32)
        if self.center:
            x[self.shape[0] // 2, self.shape[1] // 2, self.shape[2] // 2] = 1
            _num = _num-1
        _i, _j, _k = np.random.randint(self.pad_from_boundary, np.min(self.shape) - self.pad_from_boundary, (3,_num))
        x[_i, _j, _k] = 1.

        self.phantom_obj = x
        # self.check_sum_of_phantom_obj()

    def get(self):
        self.check_sum_of_phantom_obj()
        return self.phantom_obj



class Sphere(Phantoms3D):

    """
        creates 3D sphere
        :param obj_shape: tuple, object shape as (z,y,x), e.g. (64,64,64)
        :param units: tuple, voxel size in microns, e.g. (0.1,0.1,0.1)
        :param radius: scalar, radius of sphere in microns, e.g. 0.5
        :param off_centered: tuple, displacement vector by which center is moved as (k,j,i) e.g. (0,0.5,0.5)
    """

    def __init__(self, shape, units, radius, off_centered=(0,0,0)):
        super().__init__(shape)
        self.shape=shape
        self.units= units
        self.radius = radius
        self.off_centered = off_centered
        self._img = np.zeros(shape)
        self.generate()

    def generate(self):

        isinstance(self.off_centered,tuple) or _raise(ValueError("Displacement vector for center is not a 3D vector"))
        if isinstance(self.radius,(list,tuple)):
            self.radius = random.choice(self.radius)
        np.isscalar(self.radius) or _raise(ValueError("Radius has to be scalar"))
        all(2*self.radius<_u*_o for _u,_o in zip(self.units,self.shape)) or _raise(ValueError("Object diameter is bigger than object shape"))

                
        xs = list(u*(np.arange(s)-s/2) for u,s in zip(self.units, self.shape)) 
        xs = tuple(_x-self.off_centered[i] for i,_x in enumerate(xs))
        Z,Y,X = np.meshgrid(*xs,indexing = "ij")
        R = np.sqrt(X**2+Y**2+Z**2)
        mask = 1.*(R<=self.radius)

        self.phantom_obj = mask
        self.check_sum_of_phantom_obj()
    
    def get(self):
        self.check_sum_of_phantom_obj()
        return self.phantom_obj
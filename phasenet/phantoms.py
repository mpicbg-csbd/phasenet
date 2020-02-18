from csbdeep.utils import _raise
import numpy as np
import random
import warnings

def _sum(arr):
    if np.sum(arr) <= 0:
        warnings.warn("No object created")

class PhantomGenerator3D():

    """
        encapsulates Phantoms

        :param obj_shape: tuple, object shape as (z,y,x), e.g. (64,64,64)
        :param units: tuple, voxel size in microns, e.g. (0.1,0.1,0.1)
    """

    def __init__(self, phantom_shape, units):

        isinstance(phantom_shape,tuple) or _raise(ValueError("Object shape has to be a tuple"))
        isinstance(units,tuple) or _raise(ValueError("Units has to be a tuple"))
        self.phantom_shape = phantom_shape
        self.units= units


    def get_phantom_img(self, name, params):

        obj =  np.zeros(self.phantom_shape)
        if name.lower() == 'points':
            'num' in params or _raise(ValueError('Number of points not defined'))  
            num = params.get('num')
            center = params.get('center', True)
            pad_from_boundary = params.get('pad_from_boundary ', np.min(self.phantom_shape)//4)
            obj = self.points(num = num, center = center, pad_from_boundary=pad_from_boundary)

        elif name.lower() == 'sphere':
            'radius' in params or _raise(ValueError('radius not defined'))  
            radius = params.get('radius')
            off_centered = params.get('off_centered', (0,0,0))
            obj = self.sphere(radius=radius, off_centered=off_centered)

        else:
            _raise(ValueError('Phantom not found'))
        
        return obj


    def points(self, num, center = True, pad_from_boundary=0):

        """
            creates multiple points
            
            :param num: integer, number of points, e.g. 3
            :param center: boolean, whether to have a point at the center, default is False
            :param pad_from_boundary: integer, leave space between points and bouandary. Helpful for convolution, recommended size//5
        """
        np.isscalar(pad_from_boundary) and pad_from_boundary<np.min(self.phantom_shape)//2  or _raise(ValueError( "Padding from boundary has to be scalar and bouded by object size"))
        
        x = np.zeros(self.phantom_shape, np.float32)
        if center == True:
            x[self.phantom_shape[0] // 2, self.phantom_shape[1] // 2, self.phantom_shape[2] // 2] = 1
            num = num-1
        _i, _j, _k = np.random.randint(pad_from_boundary, np.min(self.phantom_shape) - pad_from_boundary, (3,num))
        x[_i, _j, _k] = 1.

        _sum(x)
        return x


    def sphere(self, radius, off_centered=(0,0,0)):

        """
            creates 3D sphere

            :param radius: scalar, radius of sphere in microns, e.g. 0.5
            :param off_centered: tuple, displacement vector by which center is moved as (k,j,i) e.g. (0,0.5,0.5)
        """

        isinstance(off_centered,tuple) or _raise(ValueError("Displacement vector for center is not a 3D vector"))
        if isinstance(radius,(list,tuple)):
            radius = random.choice(radius)
        np.isscalar(radius) or _raise(ValueError("Radius has to be scalar"))
        all(2*radius<_u*_o for _u,_o in zip(self.units,self.phantom_shape)) or _raise(ValueError("Object diameter is bigger than object shape"))

                
        xs = list(u*(np.arange(s)-s/2) for u,s in zip(self.units, self.phantom_shape)) 
        xs = tuple(_x-off_centered[i] for i,_x in enumerate(xs))
        Z,Y,X = np.meshgrid(*xs,indexing = "ij")
        R = np.sqrt(X**2+Y**2+Z**2)
        mask = 1.*(R<=radius)

        _sum(mask)

        return mask
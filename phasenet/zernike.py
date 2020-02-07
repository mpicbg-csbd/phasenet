import numpy as np
from scipy.special import binom
from csbdeep.utils import _raise
from functools import lru_cache

from abc import ABC


# Helper function (probably don't expose to users)


def nm_to_noll(n, m):
    j = (n*(n+1))//2 + abs(m)
    if m> 0 and n%4 in (0,1): return j
    if m< 0 and n%4 in (2,3): return j
    if m>=0 and n%4 in (2,3): return j+1
    if m<=0 and n%4 in (0,1): return j+1
    assert False


def nm_to_ansi(n, m):
    return (n*(n+2) + m) // 2


def nm_normalization(n, m):
    """the norm of the zernike mode n,m in born/wolf convetion

    i.e. sqrt( \int | z_nm |^2 )
    """
    return np.sqrt((1.+(m==0))/(2.*n+2))


def nm_polynomial(n, m, rho, theta, normed=True):
    """returns the zernike polyonimal by classical n,m enumeration

    if normed=True, then they form an orthonormal system

        \int z_nm z_n'm' = delta_nn' delta_mm'

        and the first modes are

        z_nm(0,0)  = 1/sqrt(pi)*
        z_nm(1,-1) = 1/sqrt(pi)* 2r cos(phi)
        z_nm(1,1)  = 1/sqrt(pi)* 2r sin(phi)
        z_nm(2,0)  = 1/sqrt(pi)* sqrt(3)(2 r^2 - 1)
        ...
        z_nm(4,0)  = 1/sqrt(pi)* sqrt(5)(6 r^4 - 6 r^2 +1)
        ...

    if normed =False, then they follow the Born/Wolf convention
        (i.e. min/max is always -1/1)

        \int z_nm z_n'm' = (1.+(m==0))/(2*n+2) delta_nn' delta_mm'

        z_nm(0,0)  = 1
        z_nm(1,-1) = r cos(phi)
        z_nm(1,1)  =  r sin(phi)
        z_nm(2,0)  = (2 r^2 - 1)
        ...
        z_nm(4,0)  = (6 r^4 - 6 r^2 +1)


    """
    if abs(m) > n:
        raise ValueError(" |m| <= n ! ( %s <= %s)" % (m, n))

    if (n - m) % 2 == 1:
        return 0 * rho + 0 * theta

    radial = 0
    m0 = abs(m)

    for k in range((n - m0) // 2 + 1):
        radial += (-1.) ** k * binom(n - k, k) * binom(n - 2 * k, (n - m0) // 2 - k) * rho ** (n - 2 * k)

    radial *= (rho <= 1.)

    if normed:
        prefac = 1. / nm_normalization(n, m)
    else:
        prefac = 1.
    if m >= 0:
        return prefac * radial * np.cos(m0 * theta)
    else:
        return prefac * radial * np.sin(m0 * theta)


@lru_cache(maxsize=32)
def rho_theta(size):
    r = np.linspace(-1,1,size)
    X,Y = np.meshgrid(r,r, indexing='ij')
    rho = np.hypot(X,Y)
    theta = np.arctan2(Y,X)
    return rho, theta


@lru_cache(maxsize=32)
def outside_mask(size):
    rho, theta = rho_theta(size)
    return nm_polynomial(0, 0, rho, theta, normed=False) < 1


# better way to do this?
def dict_to_list(kv):
    max_key = max(kv.keys())
    out = [0]*(max_key+1)
    for k,v in kv.items():
        out[k] = v
    return out


def ensure_dict(values, order):
    if isinstance(values,dict):
        return values
    if isinstance(values,np.ndarray):
        values = tuple(values.ravel())
    if isinstance(values,(tuple,list)):
        order = str(order).lower()
        order in ('noll','ansi') or _raise(ValueError())
        offset = 1 if order=='noll' else 0
        indices = range(offset,offset+len(values))
        return dict(zip(indices,values))
    raise ValueError()



# Zernike class



class Zernike:
    _ansi_names = ['piston', 'tilt', 'tip', 'oblique astigmatism', 'defocus',
                   'vertical astigmatism', 'vertical trefoil', 'vertical coma',
                   'horizontal coma', 'oblique trefoil', 'oblique quadrafoil',
                   'oblique secondary astigmatism', 'primary spherical',
                   'vertical secondary astigmatism', 'vertical quadrafoil']
    _nm_pairs = set((n,m) for n in range(200) for m in range(-n,n+1,2))
    _noll_to_nm = dict(zip((nm_to_noll(*nm) for nm in _nm_pairs),_nm_pairs))
    _ansi_to_nm = dict(zip((nm_to_ansi(*nm) for nm in _nm_pairs),_nm_pairs))

    ##############################################

    def __init__(self, index, order='noll'):
        super().__setattr__('_mutable', True)
        if isinstance(index,str):
            name = index.lower()
            name in self._ansi_names or _raise(ValueError())
            index = self._ansi_names.index(name)
            order = 'ansi'

        if isinstance(index,(list,tuple)) and len(index)==2:
            self.n, self.m = int(index[0]), int(index[1])
            (self.n, self.m) in self._nm_pairs or _raise(ValueError())
        elif isinstance(index,int):
            order = str(order).lower()
            order in ('noll','ansi') or _raise(ValueError())
            if order == 'noll':
                index in self._noll_to_nm or _raise(ValueError())
                self.n, self.m = self._noll_to_nm[index]
            elif order == 'ansi':
                index in self._ansi_to_nm or _raise(ValueError())
                self.n, self.m = self._ansi_to_nm[index]
        else:
            raise ValueError()

        self.index_noll = nm_to_noll(self.n, self.m)
        self.index_ansi = nm_to_ansi(self.n, self.m)
        self.name = self._ansi_names[self.index_ansi] if self.index_ansi < len(self._ansi_names) else None
        self._mutable = False


    def polynomial(self, size, normed=True, outside=np.nan):
        np.isscalar(size) and int(size) > 0 or _raise(ValueError())
        return self.phase(*rho_theta(int(size)), normed=normed, outside=outside)


    def phase(self, rho, theta, normed=True, outside=None):
        (isinstance(rho,np.ndarray) and rho.ndim==2 and rho.shape[0]==rho.shape[1]) or _raise(ValueError())
        (isinstance(theta,np.ndarray) and theta.shape==rho.shape) or _raise(ValueError())
        size = rho.shape[0]
        np.isscalar(normed) or _raise(ValueError())
        outside is None or np.isscalar(outside) or _raise(ValueError())
        w = nm_polynomial(self.n, self.m, rho, theta, normed=bool(normed))
        if outside is not None:
            w[outside_mask(size)] = outside
        return w


    def __hash__(self):
        return hash((self.n,self.m))


    def __eq__(self, other):
        return isinstance(other,Zernike) and (self.n,self.m) == (other.n,other.m)


    def __lt__(self, other):
        return self.index_ansi < other.index_ansi


    def __setattr__(self, *args):
        if self._mutable:
            super().__setattr__(*args)
        else:
            raise AttributeError('Zernike is immutable')


    def __repr__(self):
        return f'Zernike(n={self.n}, m={self.m: 1}, noll={self.index_noll:2}, ansi={self.index_ansi:2}' + \
               (f", name='{self.name}')" if self.name is not None else ")")



class ZernikeWavefront:
    def __init__(self, amplitudes, order='noll'):
        amplitudes = ensure_dict(amplitudes, order)
        all(np.isscalar(a) for a in amplitudes.values()) or _raise(ValueError())

        self.zernikes = {Zernike(j,order=order):a for j,a in amplitudes.items()}
        self.amplitudes_noll = tuple(dict_to_list({z.index_noll:a for z,a in self.zernikes.items()})[1:])
        self.amplitudes_ansi = tuple(dict_to_list({z.index_ansi:a for z,a in self.zernikes.items()}))
        self.amplitudes_requested = tuple(self.zernikes[k] for k in sorted(self.zernikes.keys()))


    def __len__(self):
        return len(self.zernikes)


    def polynomial(self, size, normed=True, outside=np.nan):
        return np.sum([a * z.polynomial(size=size, normed=normed, outside=outside) for z,a in self.zernikes.items()], axis=0)


    def phase(self, rho, theta, normed=True, outside=None):
        return np.sum([a * z.phase(rho=rho, theta=theta, normed=normed, outside=outside) for z,a in self.zernikes.items()], axis=0)



def random_zernike_wavefront(amplitude_ranges, order='noll', rng=None):
        if rng is None: rng = np.random
        amplitude_ranges = ensure_dict(amplitude_ranges, order)
        all((np.isscalar(v) and v>=0) or (isinstance(v,(tuple,list)) and len(v)==2) for v in amplitude_ranges.values()) or _raise(ValueError())
        amplitude_ranges = {k:((-v,v) if np.isscalar(v) else v) for k,v in amplitude_ranges.items()}
        all(v[0]<=v[1] for v in amplitude_ranges.values()) or _raise(ValueError())
        return ZernikeWavefront({k:rng.uniform(*v) for k,v in amplitude_ranges.items()}, order=order)

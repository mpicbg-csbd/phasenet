import numpy as np
from csbdeep.utils import _raise

from abc import ABC



def _nm_to_noll(n, m):
    j = (n*(n+1))//2 + abs(m)
    if m> 0 and n%4 in (0,1): return j
    if m< 0 and n%4 in (2,3): return j
    if m>=0 and n%4 in (2,3): return j+1
    if m<=0 and n%4 in (0,1): return j+1
    assert False

def _nm_to_ansi(n, m):
    return (n*(n+2) + m) // 2


class Zernike(object):
    # TODO: doesn't agree with wikpedia
    # noll = [0, 2, 1, 4, 3, 5, 8, 6, 7, 9, 14, 12, 10, 11, 13]
    # ansi = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # nm = [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2), (3, -3), (3, -1), (3, 1), (3, 3), (4, -4), (4, -2), (4, 0),
    #       (4, 2), (4, 4)]
    _ansi_names = ['piston', 'tilt', 'tip', 'oblique astigmatism', 'defocus', 'vertical astigmatism', 'vertical trefoil',
                  'vertical coma', 'horizontal coma', 'oblique trefoil', 'oblique quadrafoil',
                  'oblique secondary astigmatism', 'primary spherical', 'vertical secondary astigmatism',
                  'vertical quadrafoil']

    _nm_pairs = [(n,m) for n in range(200) for m in range(-n,n+1,2)]
    _noll_to_nm = dict(zip([_nm_to_noll(*p) for p in _nm_pairs],_nm_pairs))
    _ansi_to_nm = dict(zip([_nm_to_ansi(*p) for p in _nm_pairs],_nm_pairs))

    ##############################################

    def _get_nm(self, nm):
        try:
            if len(nm) != 2:
                return False
            n, m = nm
            if not (isinstance(n,int) and isinstance(m,int) and n >= abs(m) >= 0):
                return False
            return n, m
        except TypeError:
            return False

    def __init__(self, index, order='noll'):
        self.order = str(order).lower()
        self.order in ('noll','ansi','nm') or _raise(ValueError())
        self.order=='nm' or (isinstance(index,int) and 0 <= index) or _raise(ValueError())
        if self.order == 'nm':
            nm = self._get_nm(index)
            nm or _raise(ValueError())
            self.n, self.m = nm
        elif self.order == 'noll':
            self.n, self.m = self._noll_to_nm[index]
        elif self.order == 'ansi':
            self.n, self.m = self._ansi_to_nm[index]
        self.index_noll = _nm_to_noll(self.n, self.m)
        self.index_ansi = _nm_to_ansi(self.n, self.m)
        self.name = self._ansi_names[self.index_ansi] if self.index_ansi < len(self._ansi_names) else None

    def __repr__(self):
        return f'Zernike(n={self.n}, m={self.m: 1}, noll={self.index_noll:2}, ansi={self.index_ansi:2}' + \
               (f", name='{self.name}')" if self.name is not None else ")")




# class Wavefront(ABC):
#     pass

# class ZernikeWavefront(Wavefront):
#     def __init__(self, size, coefficients, order='noll', normed=True):
#         np.isscalar(size) and 0 < size and int(size) == size or _raise(ValueError())
#         str(order).lower() in ('noll','ansi','nm')           or _raise(ValueError())
#         np.isscalar(normed) and bool(normed) in (False,True) or _raise(ValueError())

#         self.size = int(size)
#         self.order = str(order).lower()
#         self.normed = bool(normed)

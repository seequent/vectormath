"""vector.py contains definitions for Vector and VectorArray classes"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class BaseVector(np.ndarray):
    """Class to contain basic operations used by all Vector classes"""

    def __new__(cls, *args, **kwargs):
        """BaseVector cannot be created"""
        raise NotImplementedError('Please specify Vector2 or Vector3')

    def __array_finalize__(self, obj):                                         #pylint: disable=no-self-use
        """This is called at the end of array creation

        obj depends on the context. Currently, this does not need to do
        anything regardless of context. See `subclassing docs
        <https://docs.scipy.org/numpy/user/basics.subclassing.html>`_.
        """
        if obj is None:
            return

    @property
    def x(self):
        """x-component of vector"""
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        """y-component of vector"""
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def length(self):
        """Length of vector"""
        return float(np.sqrt(np.sum(self**2)))

    @length.setter
    def length(self, value):
        if not np.isscalar(value):
            raise ValueError('Length must be a scalar')
        value = float(value)
        if self.length != 0:
            new_length = value/self.length
            self *= new_length
            return
        if value != 0:
            raise ZeroDivisionError('Cannot resize vector of length 0 to '
                                    'nonzero length')

    def as_length(self, value):
        """Return a new vector scaled to given length"""
        new_vec = self.copy()
        new_vec.length = value
        return new_vec

    def as_percent(self, value):
        """Return a new vector scaled by given decimal percent"""
        new_vec = self.copy()
        new_vec.length = value * self.length
        return new_vec

    def as_unit(self):
        """Return a new vector scaled to length 1"""
        new_vec = self.copy()
        new_vec.normalize()
        return new_vec

    def normalize(self):
        """Scale the length of a vector to 1 in place"""
        self.length = 1
        return self

    def dot(self, vec):
        """Dot product with another vector"""
        if not isinstance(vec, self.__class__):
            raise TypeError('Dot product operand must be a vector')
        return np.dot(self, vec)

    def cross(self, vec):
        """Cross product with another vector"""
        if not isinstance(vec, self.__class__):
            raise TypeError('Cross product operand must be a vector')
        return self.__class__(np.cross(self, vec))

    def __mul__(self, multiplier):
        return self.__class__(self.view(np.ndarray) * multiplier)


class Vector3(BaseVector):
    """Primitive 3D vector defined from the origin

    New Vector3 can be created with:
        - another Vector3
        - length-3 array
        - x, y, and y values
        - no input (returns [0., 0., 0.])
    """

    def __new__(cls, x=None, y=None, z=None):                                  #pylint: disable=arguments-differ

        def read_array(X, Y, Z):
            """Build Vector3 from another Vector3, [x, y, z], or x/y/z"""
            if isinstance(X, cls) and Y is None and Z is None:
                return cls(X.x, X.y, X.z)
            if (isinstance(X, (list, tuple, np.ndarray)) and len(X) == 3 and
                    Y is None and Z is None):
                return cls(X[0], X[1], X[2])
            if X is None and Y is None and Z is None:
                return cls(0, 0, 0)
            if np.isscalar(X) and np.isscalar(Y) and np.isscalar(Z):
                xyz = np.r_[X, Y, Z]
                xyz = xyz.astype(float)
                return xyz.view(cls)
            raise ValueError('Invalid input for Vector3 - must be an instance '
                             'of a Vector3, a length-3 array, 3 scalars, or '
                             'nothing for [0., 0., 0.]')

        return read_array(x, y, z)

    @property
    def z(self):
        """z-component of vector"""
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value


class Vector2(BaseVector):
    """Primitive 2D vector defined from the origin

    New Vector2 can be created with:
        - another Vector2
        - length-2 array
        - x and y values
        - no input (returns [0., 0.])
    """

    def __new__(cls, x=None, y=None):                                          #pylint: disable=arguments-differ

        def read_array(X, Y):
            """Build Vector2 from another Vector2, [x, y], or x/y"""
            if isinstance(X, cls) and Y is None:
                return cls(X.x, X.y)
            if (isinstance(X, (list, tuple, np.ndarray)) and len(X) == 2 and
                    Y is None):
                return cls(X[0], X[1])
            if X is None and Y is None:
                return cls(0, 0)
            if np.isscalar(X) and np.isscalar(Y):
                xyz = np.r_[X, Y]
                xyz = xyz.astype(float)
                return xyz.view(cls)
            raise ValueError('Invalid input for Vector2 - must be an instance '
                             'of a Vector2, a length-2 array, 2 scalars, or '
                             'nothing for [0., 0.]')

        return read_array(x, y)


class BaseVectorArray(BaseVector):
    """Class to contain basic operations used by all VectorArray classes"""

    @property
    def x(self):
        """Array of x-component of vectors"""
        return self[:, 0]

    @x.setter
    def x(self, value):
        self[:, 0] = value

    @property
    def y(self):
        """Array of y-component of vectors"""
        return self[:, 1]

    @y.setter
    def y(self, value):
        self[:, 1] = value

    @property
    def nV(self):
        """Number of vectors"""
        return self.shape[0]

    def normalize(self):
        """Scale the length of all vectors to 1 in place"""
        self.length = np.ones(self.nV)
        return self

    @property
    def dims(self):
        """Tuple of different dimension names for Vector type"""
        raise NotImplementedError('Please use Vector2Array or Vector3Array')

    @property
    def length(self):
        """Array of vector lengths"""
        return np.sqrt(np.sum(self**2, axis=1)).view(np.ndarray)

    @length.setter
    def length(self, l):
        l = np.array(l)
        if self.nV != l.size:
            raise ValueError('Length vector must be the same number of '
                             'elements as vector.')
        # This case resizes all vectors with nonzero length
        if np.all(self.length != 0):
            new_length = l/self.length
            for dim in self.dims:
                setattr(self, dim, new_length*getattr(self, dim))
            return
        # This case only applies to single vectors
        if self.nV == 1 and l == 0:
            assert self.length == 0, \
                'Nonzero length should be resized in the first case'
            for dim in self.dims:
                setattr(self, dim, 0.)
            return
        # This case only applies if vectors with length == 0
        # in an array are getting resized to 0
        if self.nV > 1 and np.array_equal(self.length.nonzero(), l.nonzero()): #pylint: disable=no-member
            new_length = l/[x if x != 0 else 1 for x in self.length]
            for dim in self.dims:
                setattr(self, dim, new_length*getattr(self, dim))
            return
        # Error if length zero array is resized to nonzero value
        raise ZeroDivisionError('Cannot resize vector of length 0 to '
                                'nonzero length')

    def dot(self, vec):
        """Dot product with another vector"""
        if not isinstance(vec, self.__class__):
            raise TypeError('Dot product operand must be a VectorArray')
        if self.nV != 1 and vec.nV != 1 and self.nV != vec.nV:
            raise ValueError('Dot product operands must have the same '
                             'number of elements.')
        return np.sum((getattr(self, d)*getattr(vec, d) for d in self.dims), 1)


class Vector3Array(BaseVectorArray):
    """List of Vector3

    A new Vector3Array can be created with:
        - another Vector3Array
        - x/y/z lists of equal length
        - n x 3 array
        - nothing (returns [[0., 0., 0.]])
    """

    def __new__(cls, x=None, y=None, z=None):                                  #pylint: disable=arguments-differ

        def read_array(X, Y, Z):
            """Build Vector3Array from various inputs"""
            if isinstance(X, cls) and Y is None and Z is None:
                X = np.atleast_2d(X)
                return cls(X.x.copy(), X.y.copy(), X.z.copy())
            if isinstance(X, (list, tuple)):
                X = np.array(X)
            if isinstance(Y, (list, tuple)):
                Y = np.array(Y)
            if isinstance(Z, (list, tuple)):
                Z = np.array(Z)
            if isinstance(X, np.ndarray) and Y is None and Z is None:
                X = np.squeeze(X)
                if X.size == 3:
                    X = X.flatten()
                    return cls(X[0], X[1], X[2])
                elif len(X.shape) == 2 and X.shape[1] == 3:
                    return cls(
                        X[:, 0].copy(), X[:, 1].copy(), X[:, 2].copy()
                    )
                raise ValueError(
                    'Unexpected shape for vector init: {shp}'.format(
                        shp=X.shape
                    )
                )
            if np.isscalar(X) and np.isscalar(Y) and np.isscalar(Z):
                X, Y, Z = float(X), float(Y), float(Z)
            elif not (isinstance(X, type(Y)) and isinstance(X, type(Z))):
                raise TypeError('Must be the same types for x, y, and '
                                'z for vector init')
            if isinstance(X, np.ndarray):
                if not (X.shape == Y.shape and X.shape == Z.shape):
                    raise ValueError('Must be the same shapes for x, y, '
                                     'and z in vector init')
                vec_ndarray = np.c_[X, Y, Z]
                vec_ndarray = vec_ndarray.astype(float)
                return vec_ndarray.view(cls)
            if X is None:
                X, Y, Z = 0.0, 0.0, 0.0
            vec_ndarray = np.r_[X, Y, Z].reshape((1, 3))
            return np.asarray(vec_ndarray).view(cls)

        return read_array(x, y, z)

    def __getitem__(self, i):
        """Overriding _getitem__ allows coersion to Vector3 or ndarray"""
        item_out = super(Vector3Array, self).__getitem__(i)
        if np.isscalar(i):
            return item_out.view(Vector3)
        if isinstance(i, slice):
            return item_out
        return item_out.view(np.ndarray)

    @property
    def z(self):
        """Array of z-component of vectors"""
        return self[:, 2]

    @z.setter
    def z(self, value):
        self[:, 2] = value

    @property
    def dims(self):
        return ('x', 'y', 'z')

    def cross(self, vec):
        """Cross product with another Vector3Array"""
        if not isinstance(vec, Vector3Array):
            raise TypeError('Cross product operand must be a Vector3Array')
        if self.nV != 1 and vec.nV != 1 and self.nV != vec.nV:
            raise ValueError('Cross product operands must have the same '
                             'number of elements.')
        return Vector3Array(np.cross(self, vec))


class Vector2Array(BaseVectorArray):
    """List of Vector2

    A new Vector2Array can be created with:
        - another Vector2Array
        - x/y lists of equal length
        - n x 2 array
        - nothing (returns [[0., 0.]])
    """

    def __new__(cls, x=None, y=None):                                          #pylint: disable=arguments-differ

        def read_array(X, Y):
            """Build Vector2Array from various inputs"""
            if isinstance(X, cls) and Y is None:
                X = np.atleast_2d(X)
                return cls(X.x.copy(), X.y.copy())
            if isinstance(X, (list, tuple)):
                X = np.array(X)
            if isinstance(Y, (list, tuple)):
                Y = np.array(Y)
            if isinstance(X, np.ndarray) and Y is None:
                X = np.squeeze(X)
                if X.size == 2:
                    X = X.flatten()
                    return cls(X[0], X[1])
                elif len(X.shape) == 2 and X.shape[1] == 2:
                    return cls(
                        X[:, 0].copy(), X[:, 1].copy()
                    )
                raise ValueError(
                    'Unexpected shape for vector init: {shp}'.format(
                        shp=X.shape
                    )
                )
            if np.isscalar(X) and np.isscalar(Y):
                X, Y = float(X), float(Y)
            elif not isinstance(X, type(Y)):
                raise TypeError('Must be the same types for x and y '
                                'for vector init')
            if isinstance(X, np.ndarray):
                if X.shape != Y.shape:
                    raise ValueError('Must be the same shapes for x and y '
                                     'in vector init')
                vec_ndarray = np.c_[X, Y]
                vec_ndarray = vec_ndarray.astype(float)
                return vec_ndarray.view(cls)
            if X is None:
                X, Y = 0.0, 0.0
            vec_ndarray = np.r_[X, Y].reshape((1, 2))
            return np.asarray(vec_ndarray).view(cls)

        return read_array(x, y)

    def __getitem__(self, i):
        """Overriding _getitem__ allows coercion to Vector2 or ndarray"""
        item_out = super(Vector2Array, self).__getitem__(i)
        if np.isscalar(i):
            return item_out.view(Vector2)
        if isinstance(i, slice):
            return item_out
        return item_out.view(np.ndarray)

    @property
    def dims(self):
        return ('x', 'y')

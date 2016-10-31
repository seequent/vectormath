"""vectormath: Vector math utilities for Python built on NumPy"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .matrix import Matrix3
from .matutils import mkvc, ndgrid, ouv_to_vec, switch_ouvz, transform_ouv
from .parallelogram import Parallelogram
from .plane import Plane
from .vector import Vector3, Vector2, Vector3Array, Vector2Array

__version__ = '0.1.0'
__author__ = '3point Science'
__license__ = 'MIT'
__copyright__ = 'Copyright 2016 3point Science'

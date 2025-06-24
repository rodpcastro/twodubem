"""
TwoDuBEM
========

TwoDuBEM solves the 2D Laplace's equation using the Boundary Element Method.

Modules
-------
Element
    This module contains classes to create boundary elements.
Boundary
    This module contains the classes to create boundaries.
Potential
    This module contains the classes to solve Laplace's equation in two dimensions.
"""

from twodubem.element import StraightConstantElement
from twodubem.boundary import PolygonalBoundary
from twodubem.potential import BEMPotential

__all__ = [
    'StraightConstantElement',
    'PolygonalBoundary',
    'BEMPotential',
]

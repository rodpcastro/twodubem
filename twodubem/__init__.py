#    ┏┓┳┓  ┳┓┏┓┳┳┓    Licensed under the MIT License
#    ┏┛┃┃┓┏┣┫┣ ┃┃┃    Copyright (c) 2025 Rodrigo Castro
#    ┗━┻┛┗┻┻┛┗┛┛ ┗    https://github.com/rodpcastro/twodubem
#                     ASCII Art (Font Tmplr) by https://patorjk.com

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
Solver
    This module contains the classes to solve Laplace's equation in two dimensions.
"""

from twodubem.element import StraightConstantElement
from twodubem.boundary import Polygon, Rectangle, Square
from twodubem.solver import Solver

__all__ = [
    'StraightConstantElement',
    'Polygon',
    'Rectangle',
    'Square',
    'Solver',
]

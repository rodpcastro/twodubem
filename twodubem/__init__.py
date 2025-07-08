#    ┏┓┳┓  ┳┓┏┓┳┳┓    Licensed under the MIT License
#    ┏┛┃┃┓┏┣┫┣ ┃┃┃    Copyright (c) 2025 Rodrigo Castro
#    ┗━┻┛┗┻┻┛┗┛┛ ┗    https://github.com/rodpcastro/twodubem
#                     ASCII Art (Font Tmplr) by https://patorjk.com

"""
TwoDuBEM
========

TwoDuBEM uses the Boundary Element Method to solve two-dimensional partial differential
equations that have been formulated as integral equations.

Modules
-------
Element
    This module contains the class to create a boundary element.
Geometry
    This module contains the classes to create polygonal boundaries.
Boundary
    This module contains the class to create a collection of polygonal boundaries.
Solver
    This module contains the class to solve boundary value problems.
"""

from twodubem.boundary import Boundary
from twodubem.laplace import Laplace
from twodubem.solver import Solver

__all__ = [
    'Boundary',
    'Laplace',
    'Solver',
]

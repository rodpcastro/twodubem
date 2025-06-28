#    ┏┓┳┓  ┳┓┏┓┳┳┓    Licensed under the MIT License
#    ┏┛┃┃┓┏┣┫┣ ┃┃┃    Copyright (c) 2025 Rodrigo Castro
#    ┗━┻┛┗┻┻┛┗┛┛ ┗    https://github.com/rodpcastro/twodubem

"""
Green
=====

This module contains the base class for Green's functions.

Classes
-------
Green
    Base class for all Green's functions.
"""


class Green:

    @staticmethod
    def eval(field, source):
        """Evaluation of the Green function given field and source points."""
        raise NotImplementedError

    def get_constant_element_influence_coefficients(
        self, element, point, return_gradients=False
    ):
        """Get influence coefficients of a constant element at a field point."""
        raise NotImplementedError

    def get_linear_element_influence_coefficients(
        self, element, point, return_gradients=False
    ):
        """Get influence coefficients of a linear element at a field point."""
        raise NotImplementedError


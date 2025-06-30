#    ┏┓┳┓  ┳┓┏┓┳┳┓    Licensed under the MIT License
#    ┏┛┃┃┓┏┣┫┣ ┃┃┃    Copyright (c) 2025 Rodrigo Castro
#    ┗━┻┛┗┻┻┛┗┛┛ ┗    https://github.com/rodpcastro/twodubem

"""
Element
=======

This module contains classes to create boundary elements.

Classes
-------
Element
    Base class for boundary elements.
LineElement
    Line segment boundary element.
"""

import numpy as np
from numpy import ndarray
from twodubem._internal import ismall


class Element:
    """Base class for boundary elements."""

    pass


class LineElement(Element):
    """Line segment boundary element.

    Parameters
    ----------
    endpoint1 : ndarray[float], shape=(2,)
        First endpoint global coordinates.
    endpoint2 : ndarray[float], shape=(2,)
        Second endpoint global coordinates.
        
    Attributes
    ----------
    endpoints : ndarray[float], shape=(2,2)
        Element endpoints' global coordinates.
    node : ndarray[float], shape=(2,)
        Midpoint between endpoints.
    length : float
        Element's length.
    tangent : ndarray[float], shape=(2,)
        Element's tangent vector.
    normal : ndarray[float], shape=(2,)
        Element's normal vector.

    Methods
    -------
    get_point_local_coordinates(point_global)
        Get point's coordinates in local system.
    get_point_global_coordinates(point_local)
        Get point's coordinates in the global system.
    get_point_distance(point_global)
        Get point's distance from the element.
    """

    def __init__(self, endpoint1, endpoint2):
        self.endpoints = np.array([endpoint1, endpoint2])
        self._set_element_properties()

    def _set_element_properties(self):
        r = self.endpoints[1] - self.endpoints[0]
        self.node = self.endpoints.mean(axis=0)
        self.length = np.linalg.norm(r)
        self.tangent = r / self.length
        self.normal = np.array([self.tangent[1], -self.tangent[0]])
    
    def get_point_local_coordinates(self, point_global):
        """Get point's coordinates in the local system.

        Parameters
        ----------
        point_global : ndarray, shape=(2,)
            Point's coordinates in the global system.

        Returns
        -------
        point_local : ndarray, shape=(2,)
            Point's coordinates in the local system.
        """

        point_relative = np.array(point_global) - self.node
        x = np.dot(point_relative, self.tangent)
        y = np.dot(point_relative, self.normal)
        point_local = np.array([x, y])

        return point_local

    def get_point_global_coordinates(self, point_local):
        """Get point's coordinates in the global system.

        Parameters
        ----------
        point_local : ndarray, shape=(2,)
            Point's coordinates in the local system.

        Returns
        -------
        point_global : ndarray, shape=(2,)
            Point's coordinates in the global system.
        """

        point_global = (
            self.node
            + point_local[0] * self.tangent
            + point_local[1] * self.normal
        )

        return point_global

    def get_point_distance(self, point_global):
        """Get point's distance from the element.
        
        Parameters
        ----------
        point_global : ndarray, shape=(2,)
            Point coordinates in the global system.

        Returns
        -------
        distance : float
            Point's distance from the element.
        """

        a = 0.5 * self.length
        x, y = self.get_point_local_coordinates(point_global)

        if np.abs(x) <= a:
            distance = np.abs(y)
        else:
            relative_to_endpoints = point_global - self.endpoints
            distance_from_endpoints = np.linalg.norm(relative_to_endpoints, axis=1)
            distance = distance_from_endpoints.min()

        return distance


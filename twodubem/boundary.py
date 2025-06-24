"""
Boundary
========

This module contains the classes to create boundaries.

Classes
-------
PolygonalBoundary
    Polygonal boundary.
"""

import numpy as np
from numpy import ndarray
from element import StraightConstantElement
from _internal import ismall


class Boundary:
    # TODO: Implement parent class. This one will also be used for type annotations.
    pass


class PolygonalBoundary(Boundary):
    # TODO: Create an alternative way of inputting the boundary geometry and conditions.
    # For complex geometries, the parametric function won't be possible.

    # TODO: Add boundary_geometry and boundary_condition function to a script that generates
    # boundary_geometry and boundary_condition arrays for a given number of elements.

    # TODO: Implement method that returns what element a point is closer to, if this
    # element is at the boundary.

    """Polygonal boundary.

    Parameters
    ----------
    boundary_geometry : callable[float]
        Parametric function that describes the boundary. The parameters must range from
        0.0 to 1.0.
    boundary_condition : callable[float, float]
        Function that returns boundary condition values for (x, y) coordinates given as
        inputs.
    number_of_elements : int
        Number of elements.

    Attributes
    ----------
    number_of_elements : int
        Number of elements.
    endpoints : ndarray[float], shape=(n, 2)
        Vertices of the polygonal boundary. If the last endpoint is not coincident
        with the first, a new endpoint, equal to the first, is created to close the
        boundary.
    elements : list[StraightConstantElement]
        List of elements.
    bc_types : ndarray[int]
        Boundary condition types by element. Value ``0`` represents Dirichlet boundary
        condition and value ``1`` represents Neumann boundary condition.
    bc_values : ndarray[float]
        Boundary condition values by element.

    Methods
    -------
    is_on_boundary(point)
        Determine if ``point`` is on the boundary.
    is_inside_region(point)
        Determine if ``point`` is inside the region enclosed by the boundary.
    show()
        Display a graphical representation of the boundary.
    """

    def __init__(self, boundary_geometry, boundary_condition, number_of_elements):
        self.number_of_elements = number_of_elements
        self._set_enpoints(boundary_geometry)
        self._set_elements()
        self._set_boundary_conditions(boundary_condition)

    def _set_enpoints(self, boundary_geometry):
        self.endpoints = np.empty((self.number_of_elements + 1, 2), dtype=np.float64)
        for i, t in enumerate(np.linspace(0.0, 1.0, self.number_of_elements + 1)):
            self.endpoints[i] = boundary_geometry(t)
    
    def _set_elements(self):
        self.elements = []
        for i in range(self.number_of_elements):
            self.elements.append(
                StraightConstantElement(
                    self.endpoints[i],
                    self.endpoints[i+1],
                )
            )

    def _set_boundary_conditions(self, boundary_condition):
        self.bc_types = np.empty(self.number_of_elements, dtype=np.int8)
        self.bc_values = np.empty(self.number_of_elements, dtype=np.float64)
        for i, element in enumerate(self.elements):
            bc_type, bc_value = boundary_condition(*element.node)
            self.bc_types[i] = bc_type
            self.bc_values[i] = bc_value

    def is_on_boundary(self, point):
        """Determine if ``point`` is on the boundary."""

        for element in self.elements:
            distance = element.get_point_distance(point)
            if ismall(distance, element.length):
                return True

        return False

    def is_inside_region(self, point):
        """Determine if ``point`` is inside the region enclosed by the boundary."""

        # Ray-casting algorithm.
        number_of_horizontal_intersections = 0
        for element in self.elements:
            endpoints_relative = element.endpoints - point
            if np.any(endpoints_relative[:, 0] >= 0):
                if endpoints_relative[:, 1].prod() < 0:
                    number_of_horizontal_intersections += 1

        if number_of_horizontal_intersections % 2 == 0:
            return False
        else:
            return True
    
    def show(self):
        """Display a graphical representation of the boundary."""

        import matplotlib.pyplot as plt

        plt.plot(
            self.endpoints[:, 0],
            self.endpoints[:, 1],
            'r-',
            marker='o',
            markersize=3,
            markerfacecolor='b',
            markeredgecolor='k',
            linewidth=2,
        )
        plt.gca().set_aspect('equal')
        plt.show()

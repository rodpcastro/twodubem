#    ┏┓┳┓  ┳┓┏┓┳┳┓    Licensed under the MIT License
#    ┏┛┃┃┓┏┣┫┣ ┃┃┃    Copyright (c) 2025 Rodrigo Castro
#    ┗━┻┛┗┻┻┛┗┛┛ ┗    https://github.com/rodpcastro/twodubem

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
from twodubem.element import StraightConstantElement
from twodubem._internal import ismall


class Polygon:

    # TODO: Implement method that returns what element a point is closer to, if this
    # element is at the boundary.

    # TODO: Find a way of creating, saving and loading a boundary region with holes.
    # class MultiplyConnectedPolygon

    """Parent class for polygonal boundaries.

    Parameters
    ----------
    file_name : str
        Input file name. In this file, each continuous boundary must be topped by a
		``#boundary`` statement. The first is always the external boundary, while the
		others are internal boundaries (holes). Every boundary is represented by data
		in 4 columns. The first two columns are coordinates x and y of a vertex. The
		third column is a integer value (0 or 1) indicating the boundary condition
		type at a boundary node. The fourth column defines the boundary condition value
		at a boundary node.

    Attributes
    ----------
    number_of_elements : int
        Number of elements.
    vertices : ndarray[float], shape=(n+1, 2)
        Vertices of the polygon.
    elements : list[StraightElement]
        List of elements.
    bc_types : ndarray[int], shape=(n,)
        Boundary condition types on boundary nodes. Value ``0`` represents Dirichlet boundary
        condition and value ``1`` represents Neumann boundary condition.
    bc_values : ndarray[float], shape=(n,)
        Boundary condition values on boundary nodes.

    Methods
    -------
    is_on_boundary(point)
        Determine if ``point`` is on the boundary.
    is_inside_region(point)
        Determine if ``point`` is inside the region enclosed by the boundary.
    save(file_name)
        Save geometry and boundary condition data to file.
    show()
        Display a graphical representation of the boundary.
    """

    def __init__(self, file_name):
        self._load(file_name)
        self.number_of_elements = len(self.vertices) - 1
        self._set_elements()

    def _set_elements(self):
        self.elements = []
        for i in range(self.number_of_elements):
            self.elements.append(
                StraightConstantElement(
                    self.vertices[i],
                    self.vertices[i+1],
                )
            )
    
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
            points_relative = element.points - point
            if np.any(points_relative[:, 0] >= 0):
                if points_relative[:, 1].prod() < 0:
                    number_of_horizontal_intersections += 1

        if number_of_horizontal_intersections % 2 == 0:
            return False
        else:
            return True

    def save(self, file_name):
        """Save boundary data to file."""

        with open(file_name, 'w') as file:
            for i in range(self.number_of_elements):
                file.write(
                    f'{self.vertices[i, 0]:.15e}    '
                    f'{self.vertices[i, 1]:.15e}    '
                    f'{self.bc_types[i]}    '
                    f'{self.bc_values[i]:.15e}\n'
                )

    def _load(self, file_name):
        """Load boundary data from file."""

        vertices = []
        bc_types = []
        bc_values = []
        with open(file_name, 'r') as file:
            for line in file:
                data_line = line.strip()
                if data_line:
                    # Skip empty lines.
                    x, y, bc_type, bc_value = data_line.split()
                    vertices.append([x, y])
                    bc_types.append(bc_type)
                    bc_values.append(bc_value)

        # The last vertex must be equal to the first to form a closed boundary.
        vertices.append(vertices[0])

        self.vertices = np.array(vertices, dtype=np.float64)
        self.bc_types = np.array(bc_types, dtype=np.int8)
        self.bc_values = np.array(bc_values, dtype=np.float64)

    def show(self):
        """Display a graphical representation of the boundary."""

        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        
        fig, ax = plt.subplots()

        external_boundary = Polygon(self.vertices, color='silver')
        # internal_boundary = Polygon(self.vertices, color='white')
        
        ax.add_patch(external_boundary)
        # ax.add_patch(internal_boundary)
        ax.set_aspect('equal')
        ax.plot(
            self.vertices[:, 0],
            self.vertices[:, 1],
            'r-',
            marker='o',
            markersize=3,
            markerfacecolor='b',
            markeredgecolor='k',
            linewidth=2,
        )
        plt.show()

    def _copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def __neg__(self):
        negative_polygon = self._copy()
        negative_polygon.vertices = self.vertices[::-1]
        negative_polygon.bc_types = self.bc_types[::-1]
        negative_polygon.bc_values = self.bc_values[::-1]
        negative_polygon._set_elements()
        
        return negative_polygon


class Rectangle(Polygon):
    """Rectangular boundary.

    Parameters
    ----------
    """

    def __init__(
        self,
        bottom_left_corner,
        width,
        height,
        number_of_width_elements,
        number_of_height_elements,
        boundary_condition,
    ):
        self.number_of_elements = 2 * (number_of_width_elements 
                                       + number_of_height_elements)
        self._set_vertices(
            bottom_left_corner,
            width,
            height,
            number_of_width_elements,
            number_of_height_elements,
        )
        self._set_elements()
        self._set_boundary_conditions(
            number_of_width_elements,
            number_of_height_elements,
            boundary_condition,
        )

    def _set_vertices(self, p0, w, h, nx, ny):
        self.vertices = np.empty((self.number_of_elements + 1, 2), dtype=np.float64)
        for side in range(4):
            if side == 0:
                # Bottom side.
                x0 = p0[0]
                x1 = p0[0] + w
                i0 = 0
                i1 = nx
                self.vertices[i0:i1, 0] = np.linspace(x0, x1, nx, endpoint=False)
                self.vertices[i0:i1, 1] = p0[1]
            elif side == 1:
                # Right side.
                y0 = p0[1]
                y1 = p0[1] + h
                i0 = nx
                i1 = nx + ny
                self.vertices[i0:i1, 0] = p0[0] + w
                self.vertices[i0:i1, 1] = np.linspace(y0, y1, ny, endpoint=False)
            elif side == 2:
                # Top side.
                x0 = p0[0] + w
                x1 = p0[0]
                i0 = nx + ny
                i1 = nx + ny + nx
                self.vertices[i0:i1, 0] = np.linspace(x0, x1, nx, endpoint=False)
                self.vertices[i0:i1, 1] = p0[1] + h
            elif side == 3:
                # Left side.
                y0 = p0[1] + h
                y1 = p0[1]
                i0 = nx + ny + nx
                i1 = nx + ny + nx + ny
                self.vertices[i0:i1, 0] = p0[0]
                self.vertices[i0:i1, 1] = np.linspace(y0, y1, ny, endpoint=False)

        self.vertices[-1] = self.vertices[0]

    def _set_boundary_conditions(self, nx, ny, boundary_condition):
        self.bc_types = np.empty(self.number_of_elements, dtype=np.int8)
        self.bc_values = np.empty(self.number_of_elements, dtype=np.float64)
        for i, element in enumerate(self.elements):
            if 0 <= i < nx:
                side = 0
            elif nx <= i < nx + ny:
                side = 1
            elif nx + ny <= i < nx + ny + nx:
                side = 2
            elif nx + ny + nx <= i < nx + ny + nx + ny:
                side = 3

            bc_type, bc_value = boundary_condition(side, element.node)
            self.bc_types[i] = bc_type
            self.bc_values[i] = bc_value


class Square(Rectangle):
    """Square boundary."""

    def __init__(
        self,
        bottom_left_corner,
        side_length,
        number_of_side_elements,
        boundary_condition,
    ):
        super().__init__(
            bottom_left_corner,
            side_length,
            side_length,
            number_of_side_elements,
            number_of_side_elements,
            boundary_condition,
        )

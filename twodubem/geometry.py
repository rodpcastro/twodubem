
#    ┏┓┳┓  ┳┓┏┓┳┳┓    Licensed under the MIT License
#    ┏┛┃┃┓┏┣┫┣ ┃┃┃    Copyright (c) 2025 Rodrigo Castro
#    ┗━┻┛┗┻┻┛┗┛┛ ┗    https://github.com/rodpcastro/twodubem

"""
Geometry
========

This module contains the classes to create polygonal boundaries.

Classes
-------
Polygon
    Polygonal boundary.
Rectangle
    Rectangular boundary.
Square
    Square boundary.
Circle
    Circular boundary. Not implemented.
"""

import numpy as np
from numpy import ndarray
from twodubem.element import LineElement
from twodubem._internal import ismall, eps


class Polygon:
    """Polygonal boundary.

    Objects of this class define a boundary by a simply connected polygon. The domain
    is located inside or outside the boundary, depending on the orientation of the
    vertices, which also define the directions of the boundary normal and tangent
    vectors.

    The normal vector always points outwards from the domain, and the tangent vector's
    direction follows the boundary orientation (counterclockwise or clockwise).

    If the vertices of a boundary are ordered in a counterclockwise orientation, the 
    domain lies inside the boundary. Conversely, if the vertices are ordered in a
    clockwise orientation, the domain lies outside the boundary.

    Parameters
    ----------
    vertices : ndarray[float], shape=(n+1, 2)
        Vertices of the polygon. On each boundary, the last vertex is coincident with
        the first to form a closed boundary.
    boundary_condition_types : ndarray[int], shape(n,)
        Boundary condition types on boundary nodes. Value ``0`` represents Dirichlet
        boundary condition and value ``1`` represents Neumann boundary condition.
    boundary_condition_values : ndarray[float], shape=(n,)
        Boundary condition values on boundary nodes.

    Attributes
    ----------
    number_of_elements : int
        Number of elements.
    vertices : ndarray[float], shape=(n+1, 2)
        Vertices of the polygon. On each boundary, the last vertex is coincident with
        the first to form a closed boundary.
    elements : list[LineElement]
        List of elements.
    bc_types : ndarray[int], shape(n,)
        Boundary condition types on boundary nodes. Value ``0`` represents Dirichlet
        boundary condition and value ``1`` represents Neumann boundary condition.
    bc_values : ndarray[float], shape=(n,)
        Boundary condition values on boundary nodes.
    bc_dirichlet : ndarray[bool], shape=(n,)
        Boolean array indicating what elements contain a Dirichlet boundary condition.
    bc_neumann : ndarray[bool], shape=(n,)
        Boolean array indicating what elements contain a Neumann boundary condition.

    Methods
    -------
    is_on_boundary(point)
        Determine if ``point`` is on the boundary.
    is_on_domain_interior(point)
        Determine if ``point`` is on the domain's interior.
    """

    def __init__(self, vertices, boundary_condition_types, boundary_condition_values):
        self.vertices = vertices
        self.bc_types = boundary_condition_types
        self.bc_values = boundary_condition_values
        self._set_elements()
        self._set_boundary_conditions()
        self._set_boundary_orientation()

    def _set_elements(self):
        self.elements = []
        for i in range(len(self.vertices) - 1):
            self.elements.append(
                LineElement(
                    self.vertices[i],
                    self.vertices[i+1],
                )
            )

        self.number_of_elements = len(self.elements)

    def _set_boundary_conditions(self):
        self.bc_neumann = np.array(self.bc_types, dtype=np.bool)
        self.bc_dirichlet = ~self.bc_neumann

    def _set_boundary_orientation(self):
        """Determine if boundary is oriented counterclockwise (+) or clockwise (-)"""

        A = 0.0
        for element in self.elements:
            x1, y1 = element.endpoints[0]
            x2, y2 = element.endpoints[1]
            A += 0.5 * (x1 * y2 - y1 * x2)

        self.orientation = np.sign(A)

    def is_on_boundary(self, point):
        """Determine if ``point`` is on the boundary.
        
        Parameters
        ----------
        point : ndarray[float], shape=(2,)
            Point coordinates in the global system.

        Returns
        -------
        on_boundary : bool
            If ``True``, ``point`` is on the boundary.
        element_index : int
            Index of the closest element. If ``point`` is not on the boundary, ``None``
            is returned. If point is on a vertex, the lowest element index is returned.
        """

        on_boundary = False
        element_index = None        
        for i, element in enumerate(self.elements):
            if element.is_on_element(point):
                on_boundary = True
                element_index = i
                break

        return on_boundary, element_index

    def is_on_domain_interior(self, point):
        """Determine if ``point`` is on the domain's interior."""

        # Ray-casting algorithm.
        on_boundary = False
        number_of_horizontal_intersections = 0
        xv = self.elements[-1].endpoints[1, 0]
        for i, element in enumerate(self.elements):
            if element.is_on_element(point):
                on_boundary = True
                break

            endpoints_relative = element.endpoints - point
            if np.any(endpoints_relative[:, 0] >= 0):
                x1, y1 = endpoints_relative[0]
                x2, y2 = endpoints_relative[1]
                if ismall(y1, element.length) and ismall(y2, element.length):
                    # Skip if the point is colinear with element's endpoints.
                    continue
                elif ismall(y1, element.length) or ismall(y2, element.length):
                    # If the point's y-coordinate is the same as a boundary 
                    # vertex's, the point is moved upward a small quantity ε.
                    endpoints_relative[:, 1] -= eps
                    x1, y1 = endpoints_relative[0]
                    x2, y2 = endpoints_relative[1]

                if endpoints_relative[:, 1].prod() < 0:
                    if np.all(endpoints_relative[:, 0] >= 0):
                        number_of_horizontal_intersections += 1
                    else:
                        x_intersection = x1 - (x2 - x1) / (y2 - y1) * y1
                        if x_intersection > 0:
                            number_of_horizontal_intersections += 1

        if not on_boundary:
            if number_of_horizontal_intersections % 2 == 0:
                return bool(self.orientation < 0)
            else:
                return bool(self.orientation > 0)
        else:
            return False

    def show(self, filename=''):
        """Display a graphical representation of the boundary.
        
        Paramaters
        ----------
        filename : str, default=''
            Name of the file in which the figure is saved. If not specified, the figure
            is not saved to a file.
        """

        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as PolygonPatch

        fig, ax = plt.subplots()

        ax.set_aspect('equal')

        if self.orientation > 0:
            color = 'silver'
        else:
            ax.set_facecolor('silver')
            color = 'white'
        
        bpatch = PolygonPatch(self.vertices, color=color)
        ax.add_patch(bpatch)
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

        if filename:
            plt.savefig(filename, bbox_inches='tight')

        plt.show()

    def __neg__(self):
        return Polygon(
            self.vertices[::-1],
            self.bc_types[::-1],
            self.bc_values[::-1],
        )

    def __sub__(self, other):
        return self.__add__(-other)

    def __add__(self, other):
        from twodubem import Boundary

        if isinstance(other, Polygon):
            boundaries = [self, other]
        elif isinstance(other, Boundary):
            boundaries = [self] + other.boundaries
        else:
            raise ValueError(f"Operand must be a Polygon or a Boundary")

        B = Boundary()
        B.boundaries = boundaries
        B._set_boundary_properties()

        return B


class Rectangle(Polygon):
    """Rectangular boundary.

    Parameters
    ----------
    bottom_left_corner : ndarray[float], shape=(2,)
        Array containing the global coordinates of the rectangle bottom left corner.
    width : float
        Rectangle width.
    height : float
        Rectangle height.
    number_of_width_elements : int
        Number of elements on bottom and top sides of the rectangle.
    number_of_height_elements : int
        Number of elements on left and right sides of the rectangle.
    boundary_condition : callable
        Function that describes the boundary condition. This function must receive as
		inputs an integer representing the side of the rectangle (bottom=0, right=1,
        top=2, left=3), and an array indicating the point where the boundary condition
        is evaluated. This functions must return the boundary condition type, ``0`` for
        Dirichlet and ``1`` for Neumann, and the boundary condition value.

    Examples
    --------
    The code below creates a 2x1 rectangle, with bottom left corner at the origin, 4
    elements on the bottom and top sides, 2 elements on the right and left sides, and
    the following boundary conditions:

    * Bottom and top sides: Dirichlet boundary condition, phi(x, y) = x * y
    * Rigth and left sides: Neumann boundary condition, q(x, y) = 0.0
    
    ```python
    from twodubem import Rectangle
    def boundary_condition(side, point):
        x, y = point
        if side in [0, 2]:
            bc_type = 0
            bc_value = x * y
        elif side in [1, 3]:
            bc_type = 1
            bc_value = 0.0
        else:
            raise ValueError('Invalid input.')
        return bc_type, bc_value
    
    R = Rectangle([0.0, 0.0], 2.0, 1.0, 4, 2, boundary_condition)
    ```
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
        super()._set_boundary_conditions()
        self._set_boundary_orientation()

    def _set_vertices(self, p0, w, h, nx, ny):
        n = 2 * (nx + ny)
        self.vertices = np.empty((n + 1, 2), dtype=np.float64)
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

        # The last vertex must be equal to the first to form a closed boundary.
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
    """Square boundary.

    This class is a simplification of ``Rectangle`` to create square boundaries.

    Parameters
    ----------
    bottom_left_corner : ndarray[float], shape=(2,)
        Array containing the global coordinates of the square bottom left corner.
    side_length : float
        Side length of the square.
    number_of_side_elements : int
        Number of elements on each side of the square.
    boundary_condition : callable
        Function that describes the boundary condition. This function must receive as
		inputs an integer representing the side of the rectangle (bottom=0, right=1,
        top=2, left=3), and an array indicating the point where the boundary condition
        is evaluated. This functions must return the boundary condition type, ``0`` for
        Dirichlet and ``1`` for Neumann, and the boundary condition value.
    """

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


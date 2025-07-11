
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
    Circular boundary.
"""

import numpy as np
from numpy import ndarray
from twodubem.element import LineElement
from twodubem._internal import eps, ismall, tozero


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
        List of line segment elements that make up the sides of the polygon.
    midpoints : ndarray[float], shape=(n, 2)
        Array containing the positions of the midpoints between vertices.
    tangents : ndarray[float], shape=(n, 2)
        Tangent vectors to the sides of the polygon.
    normals : ndarray[float], shape=(n, 2)
        Normal vectors to the sides of the polygon.
    lengths : ndarray[float], shape=(n, 2)
        Lengths of the sides of the polygon.
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
    show(filename='', show_element_index=False)
        Display a graphical representation of the boundary.
    """

    def __init__(self, vertices, boundary_condition_types, boundary_condition_values):
        self.vertices = vertices
        self.bc_types = boundary_condition_types
        self.bc_values = boundary_condition_values
        self._set_elements()
        self._set_sides_properties()
        self._set_boundary_conditions()
        self._set_boundary_orientation()
        self._set_boundary_size()

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

    def _set_sides_properties(self):
        self.midpoints = np.empty((self.number_of_elements, 2))
        self.tangents = np.empty((self.number_of_elements, 2))
        self.normals = np.empty((self.number_of_elements, 2))
        self.lengths = np.empty(self.number_of_elements)
        for i, element in enumerate(self.elements):
            self.midpoints[i] = element.node
            self.tangents[i] = element.tangent
            self.normals[i] = element.normal
            self.lengths[i] = element.length

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

    def _set_boundary_size(self):
        """Determine maximum width and height of the boundary."""

        self._lx = np.max(self.vertices[:, 0]) - np.min(self.vertices[:, 0])
        self._ly = np.max(self.vertices[:, 1]) - np.min(self.vertices[:, 1])

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

    def show(self, filename='', show_element_index=False):
        """Display a graphical representation of the boundary.
        
        Paramaters
        ----------
        filename : str, default=''
            Name of the file in which the figure is saved. If not specified, the figure
            is not saved to a file.
        show_element_index : bool, default=False
            If ``True``, the element index is displayed near the element.
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

        if show_element_index:        
            lref = np.min([self._lx, self._ly])
            index_offset = 0.03 * lref
            
            for i, element in enumerate(self.elements):
                index_position = element.node - index_offset * element.normal
                ax.text(
                    *index_position, i, ha='center', va='center', fontsize='x-small'
                )

        plt.show()

    def rotate(self, angle, center=np.zeros(2)):
        """Rotate boundary around a center point by an angle.

        Parameters
        ----------
        angle : float
            Rotation angle in radians.
        center : ndarray[float], shape=(2,), default=ndarray([0.0, 0.0])
            Rotation center point, given as an array with its global coordinates.
        """

        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])

        vertices_relative = self.vertices - center
        vertices_relative_rotated = R @ vertices_relative.T
        vertices_rotated = vertices_relative_rotated.T + center
        
        self.vertices = tozero(vertices_rotated)
        self._set_elements()

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
    from twodubem.geometry import Rectangle
    def boundary_condition(side, point):
        x, y = point
        if side in [0, 2]:
            bc_type = 0
            bc_value = x * y
        elif side in [1, 3]:
            bc_type = 1
            bc_value = 0.0
        else:
            raise ValueError('Invalid input for boundary condition')

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
        self._set_sides_properties()
        self._set_boundary_conditions(
            number_of_width_elements,
            number_of_height_elements,
            boundary_condition,
        )
        super()._set_boundary_conditions()
        self._set_boundary_orientation()
        self._set_boundary_size()

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


class Circle(Polygon):
    """Circular boundary.

    Objects of this class approximate a circle or a circular arc using polygons.

    Parameters
    ----------
    center : ndarray[float], shape=(2,)
        Array containing the global coordinates of the circle center.
    radius : float
        Circle radius.
    angle1 : float, default=0.0
        Angle in radians indicating the angular position of the first vertex.
    angle2 : float, default=2π
        Angle in radians indicating the angular position of the last vertex.
    number_of_radius_elements : int, default=0
        Number of elements along the radius. If the circle is incomplete and this
        parameter is not given, the number of elements along the radius is calculated
        such that the element's length along the radius approximate the element's
        length along the circumference.
    number_of_circumference_elements : int
        Number of elements along the circumference.
    boundary_condition : callable
        Procedure that describes the boundary condition as function of the point on
        the boundary. This functions must return the boundary condition type, ``0``
        for Dirichlet and ``1`` for Neumann, and the boundary condition value.

    Examples
    --------
    The code below creates a quarter of a circle of radius 1.0, centered at the origin,
    with 5 elements along the arc of circumference, 3 elements along the radius, and
    the following boundary conditions:

    * Bottom side: Neumann boundary condition, phi(x, y=0.0) = -1.0
    * Left side: Dirichlet boundary condition, phi(x=0.0, y) = y
    * Arc of circumference: Dirichlet boundary condition, q(x>0.0, y>0.0) = x + y
    
    ```python
    from numpy import pi
    from twodubem.geometry import Circle

    def boundary_condition(point):
        x, y = point
        if y == 0:
            bc_type = 1
            bc_value = -1.0
        elif x == 0:
            bc_type = 0
            bc_value = y
        elif x > 0 and y > 0:
            bc_type = 0
            bc_value = x + y
        else:
            raise ValueError(f"Invalid input for boundary condition")

        return bc_type, bc_value

    C = Circle([0.0, 0.0], 1.0, 5, boundary_condition, 3, 0.0, pi/2)
    ```
    """

    def __init__(
        self,
        center,
        radius,
        number_of_circumference_elements,
        boundary_condition,
        number_of_radius_elements=0,
        angle1=0.0,
        angle2=2*np.pi,
    ):
        self._set_vertices(
            center,
            radius,
            angle1,
            angle2,
            number_of_radius_elements,
            number_of_circumference_elements,
        )
        self._set_elements()
        self._set_sides_properties()
        self._set_boundary_conditions(boundary_condition)
        super()._set_boundary_conditions()
        self._set_boundary_orientation()
        self._set_boundary_size()

    def _set_vertices(self, c, r, t1, t2, nr, nc):
        vertices = []
        if ismall(t2 - t1 - 2*np.pi):
            for t in np.linspace(t1, t2, nc, endpoint=False):
                x = c[0] + r * np.cos(t)
                y = c[1] + r * np.sin(t)
                vertices.append([x, y])
        elif t2 - t1 < 2*np.pi:
            # In case the number of radial elements is not given.
            if nr == 0:
                lc = r * (t2 - t1) / nc  # Length of circumferential element.
                nr = np.round(r / lc).astype(np.int32)

            # Along the radius for angle1.
            for s in np.linspace(0.0, r, nr, endpoint=False):
                x = c[0] + s * np.cos(t1)
                y = c[1] + s * np.sin(t1)
                vertices.append([x, y])

            # Along the circumference.
            for t in np.linspace(t1, t2, nc, endpoint=False):
                x = c[0] + r * np.cos(t)
                y = c[1] + r * np.sin(t)
                vertices.append([x, y])

            # Along the radius for angle2.
            for s in np.linspace(r, 0.0, nr, endpoint=False):
                x = c[0] + s * np.cos(t2)
                y = c[1] + s * np.sin(t2)
                vertices.append([x, y])
        else:
            raise ValueError(f"angle2 - angle1 must be lower than or equal to 2π")

        # The last vertex must be equal to the first to form a closed boundary.
        vertices.append(vertices[0])

        vertices = np.array(vertices, dtype=np.float64)
        vertices = tozero(vertices)
        self.vertices = vertices

    def _set_boundary_conditions(self, boundary_condition):
        self.bc_types = np.empty(self.number_of_elements, dtype=np.int8)
        self.bc_values = np.empty(self.number_of_elements, dtype=np.float64)
        for i, element in enumerate(self.elements):
            bc_type, bc_value = boundary_condition(element.node)
            self.bc_types[i] = bc_type
            self.bc_values[i] = bc_value


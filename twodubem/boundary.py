#    ┏┓┳┓  ┳┓┏┓┳┳┓    Licensed under the MIT License
#    ┏┛┃┃┓┏┣┫┣ ┃┃┃    Copyright (c) 2025 Rodrigo Castro
#    ┗━┻┛┗┻┻┛┗┛┛ ┗    https://github.com/rodpcastro/twodubem

"""
Boundary
========

This module contains the classes to create boundaries.

Classes
-------
Polygon
    Polygonal boundary.
Rectangle
    Rectangular boundary.
Square
    Square boundary.
"""

import numpy as np
from numpy import ndarray
from twodubem.element import LineElement
from twodubem._internal import ismall


class Polygon:
    # TODO: Update docstrings to explain that the normal vector should point outward the domain.
    # TODO: If there's only one boundary, with domain on the outside, change background color to silver.
    """Polygonal boundary.
    
    See examples section for instructions on how to structure the input file.

    Parameters
    ----------
    file_name : str
        Input file name.

    Attributes
    ----------
    number_of_elements : int
        Number of elements.
    vertices : dict
        Vertices of the polygon. On each boundary, the last vertex is coincident with
        the first to form a closed boundary.
    elements : list[LineElement]
        List of elements.
    bc_types : dict
        Boundary condition types on boundary nodes. Value ``0`` represents Dirichlet
        boundary condition and value ``1`` represents Neumann boundary condition.
    bc_values : dict
        Boundary condition values on boundary nodes.

    Methods
    -------
    is_on_boundary(point)
        Determine if ``point`` is on the boundary.
    is_on_domain_interior(point)
        Determine if ``point`` is on the domain's interior.
    save(file_name)
        Save geometry and boundary condition data to file.
    show()
        Display a graphical representation of the boundary.

    Examples
    --------
    Input files must contain a header ``!boundary`` indicating the start of a boundary
    dataset. After that, numerical data is distributed in four columns, representing:

    vertices_x, vertices_y, boundary_condition_types, boundary_condition_values

    Any comments in the input file must start with ``#``. Below, there's an example of 
    input file, representing a square of unitary side with dirichlet boundary condition
    on the bottom and top sides and neumann boundary condition on the right and left
    sides.

    #input_file_start
    !boundary 0
    0.0    0.0    0    0.5
    1.0    0.0    1    0.0
    1.0    1.0    0    0.5
    0.0    1.0    1    0.0
    #input_file_end

    The vertices defining the external boundary must be ordered in the counterclockwise
    orientation. Additional boundaries in the input file are used to represent internal
    boundaries (holes), and must be defined after the external boundary. Internal
    boundaries must be ordered in the clockwise orientation.
    """

    def __init__(self, file_name):
        self._load(file_name)
        self._set_elements()
        self._set_boundary_conditions()

    def _set_elements(self):
        self.elements = []
        for i, vertices in self.vertices.items():
            for j in range(len(vertices) - 1):
                self.elements.append(
                    LineElement(
                        vertices[j],
                        vertices[j+1],
                    )
                )

        self.number_of_elements = len(self.elements)

    def _set_boundary_conditions(self):
        self.bc_neumann = np.array([], dtype=np.bool)
        self.bc_values_ = np.array([], dtype=np.float64)
        for i in self.bc_types.keys():
            self.bc_neumann = np.concatenate(
                (self.bc_neumann, self.bc_types[i].astype(np.bool))
            )
            self.bc_values_ = np.concatenate(
                (self.bc_values_, self.bc_values[i])
            )

        self.bc_dirichlet = ~self.bc_neumann
    
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
            is returned.
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
        number_of_horizontal_intersections = 0
        for element in self.elements:
            endpoints_relative = element.endpoints - point
            if np.any(endpoints_relative[:, 0] >= 0):
                if ismall(endpoints_relative[:, 1].prod(), element.length):
                    # If the point y-coordinate coincides with the polygon vertex.
                    yr1 = endpoints_relative[0, 1]
                    yr2 = endpoints_relative[1, 1]
                    if ismall(yr1, element.length) and ismall(yr2, element.length):
                        # Skip if point is colinear with element's endpoints.
                        continue
                    elif ismall(yr2, element.length):
                        # Count for only one of the elements that share the vertex.
                        number_of_horizontal_intersections += 1
                if endpoints_relative[:, 1].prod() < 0:
                    if np.all(endpoints_relative[:, 0] >= 0):
                        number_of_horizontal_intersections += 1
                    else:
                        x1, y1 = element.endpoints[0]
                        x2, y2 = element.endpoints[1]
                        x_intersection = x1 + (x2 - x1) / (y2 - y1) * (point[1] - y1)
                        if point[0] < x_intersection:
                            number_of_horizontal_intersections += 1

        if number_of_horizontal_intersections % 2 == 0:
            return False
        else:
            return True
    
    def save(self, file_name):
        """Save boundary data to file."""

        with open(file_name, 'w') as file:
            # File header.
            number_of_boundaries = max(self.vertices.keys()) + 1
            file.write(f'#TwoDuBEM\n')
            file.write(f'#Number of boundaries: {number_of_boundaries}\n')
            file.write(
                f'#Columns: '
                f'vertices_x, '
                f'vertices_y, '
                f'boundary_condition_types, '
                f'boundary_condition_values\n\n'
            )

            # File data.
            for i in self.vertices.keys():
                file.write(f'!boundary {i}\n')
                for j in range(len(self.vertices[i]) - 1):
                    file.write(
                        f'{self.vertices[i][j, 0]:.15e}    '
                        f'{self.vertices[i][j, 1]:.15e}    '
                        f'{self.bc_types[i][j]}    '
                        f'{self.bc_values[i][j]:.15e}\n'
                    )
                if i < number_of_boundaries - 1:
                    file.write(f'\n')

    def _load(self, file_name):
        """Load boundary data from file."""
        
        vertices = dict()
        bc_types = dict()
        bc_values = dict()
        with open(file_name, 'r') as file:
            i = 0
            for line in file:
                data_line = line.strip()
                if data_line:
                    # Skip empty lines.
                    if '!boundary' in data_line:
                        # Boundary data.
                        i += 1
                        vertices[i-1] = []
                        bc_types[i-1] = []
                        bc_values[i-1] = []
                        continue
                    elif i > 0:
                        x, y, bc_type, bc_value = data_line.split()
                        vertices[i-1].append([x, y])
                        bc_types[i-1].append(bc_type)
                        bc_values[i-1].append(bc_value)
        
            for j in range(i):
                # The last vertex must be equal to the first to form a closed boundary.
                vertices[j].append(vertices[j][0])
        
                # converting data to numpy arrays.
                vertices[j] = np.array(vertices[j], np.float64)
                bc_types[j] = np.array(bc_types[j], dtype=np.int8)
                bc_values[j] = np.array(bc_values[j], dtype=np.float64)

        self.vertices = vertices
        self.bc_types = bc_types
        self.bc_values = bc_values

    def show(self):
        """Display a graphical representation of the boundary."""

        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        fig, ax = plt.subplots()

        for i, vertices in self.vertices.items():
            color = 'silver' if i == 0 else 'white'
            boundary = Polygon(vertices, color=color)
            ax.add_patch(boundary)
            ax.set_aspect('equal')
            ax.plot(
                vertices[:, 0],
                vertices[:, 1],
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

    def _set_vertices(self, p0, w, h, nx, ny):
        n = 2 * (nx + ny)
        self.vertices = {0: np.empty((n + 1, 2), dtype=np.float64)}
        for side in range(4):
            if side == 0:
                # Bottom side.
                x0 = p0[0]
                x1 = p0[0] + w
                i0 = 0
                i1 = nx
                self.vertices[0][i0:i1, 0] = np.linspace(x0, x1, nx, endpoint=False)
                self.vertices[0][i0:i1, 1] = p0[1]
            elif side == 1:
                # Right side.
                y0 = p0[1]
                y1 = p0[1] + h
                i0 = nx
                i1 = nx + ny
                self.vertices[0][i0:i1, 0] = p0[0] + w
                self.vertices[0][i0:i1, 1] = np.linspace(y0, y1, ny, endpoint=False)
            elif side == 2:
                # Top side.
                x0 = p0[0] + w
                x1 = p0[0]
                i0 = nx + ny
                i1 = nx + ny + nx
                self.vertices[0][i0:i1, 0] = np.linspace(x0, x1, nx, endpoint=False)
                self.vertices[0][i0:i1, 1] = p0[1] + h
            elif side == 3:
                # Left side.
                y0 = p0[1] + h
                y1 = p0[1]
                i0 = nx + ny + nx
                i1 = nx + ny + nx + ny
                self.vertices[0][i0:i1, 0] = p0[0]
                self.vertices[0][i0:i1, 1] = np.linspace(y0, y1, ny, endpoint=False)

        # The last vertex must be equal to the first to form a closed boundary.
        self.vertices[0][-1] = self.vertices[0][0]

    def _set_boundary_conditions(self, nx, ny, boundary_condition):
        self.bc_types = {0: np.empty(self.number_of_elements, dtype=np.int8)}
        self.bc_values = {0: np.empty(self.number_of_elements, dtype=np.float64)}
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
            self.bc_types[0][i] = bc_type
            self.bc_values[0][i] = bc_value


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


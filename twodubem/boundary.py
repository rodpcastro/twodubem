#    ┏┓┳┓  ┳┓┏┓┳┳┓    Licensed under the MIT License
#    ┏┛┃┃┓┏┣┫┣ ┃┃┃    Copyright (c) 2025 Rodrigo Castro
#    ┗━┻┛┗┻┻┛┗┛┛ ┗    https://github.com/rodpcastro/twodubem

"""
Boundary
========

This module contains the class to create a collection of polygonal boundaries.

Classes
-------
Boundary
    Collection of polygonal boundaries.
"""

import numpy as np
from twodubem.geometry import Polygon
from twodubem._internal import ismall


class Boundary:
    """Collection of polygonal boundaries.

    An object of this class can be created from an input file or by operations between
    Polygon objects.

    See the examples section for instructions on how to create a Boundary object.

    Parameters
    ----------
    filename : str, default=''
        Input file name.

    Attributes
    ----------
    number_of_elements : int
        Number of elements.
    elements : list[LineElement]
        List of elements.
    boundaries : list[Polygon]
        List of boundaries.
    midpoints : ndarray[float], shape=(n, 2)
        Positions of the midpoints between vertices of the polygons that make up the
        boundary.
    tangents : ndarray[float], shape=(n, 2)
        Tangent vectors to the sides of the polygons that make up the boundary.
    normals : ndarray[float], shape=(n, 2)
        Normal vectors to the sides of the polygons that make up the boundary.
    lengths : ndarray[float], shape=(n, 2)
        Lengths of the sides of the polygons that make up the boundary.
    bc_values : dict
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
    save(filename)
        Save geometry and boundary condition data to file.
    show(filename='', show_element_index=False)
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

    The normal vector always points outwards from the domain, and the tangent vector's
    direction follows the boundary orientation (counterclockwise or clockwise).

    If the vertices of a boundary are ordered in a counterclockwise orientation, the 
    domain lies inside the boundary. Conversely, if the vertices are ordered in a
    clockwise orientation, the domain lies outside the boundary.

    A single clockwise oriented-boundary defines a domain that covers the entire plane,
    excluding the region enclosed by the boundary. A clockwise-oriented boundary
    enclosed by a counterclockwise-oriented boundary represents an internal hole.
    """

    def __init__(self, filename=''):
        self.boundaries: list[Polygon] = []
        if filename:
            self._filename = filename
            self._load(filename)
            self._set_boundary_properties()

    def _set_boundary_properties(self):
        self._set_elements()
        self._set_boundary_conditions()
        self._set_boundary_size()
        self._set_sides_properties()

    def _set_elements(self):
        self.elements = []
        for boundary in self.boundaries:
            self.elements += boundary.elements

        self.number_of_elements = len(self.elements)

    def _set_sides_properties(self):
        self.midpoints = np.empty((0, 2))
        self.tangents = np.empty((0, 2))
        self.normals = np.empty((0, 2))
        self.lengths = np.empty(0)
        for boundary in self.boundaries:
            self.midpoints = np.vstack((self.midpoints, boundary.midpoints))
            self.tangents = np.vstack((self.tangents, boundary.tangents))
            self.normals = np.vstack((self.normals, boundary.normals))
            self.lengths = np.concatenate((self.lengths, boundary.lengths))

    def _set_boundary_conditions(self):
        self.bc_neumann = np.array([], dtype=np.bool)
        self.bc_values = np.array([], dtype=np.float64)
        for boundary in self.boundaries:
            self.bc_neumann = np.concatenate(
                (self.bc_neumann, boundary.bc_neumann)
            )
            self.bc_values = np.concatenate(
                (self.bc_values, boundary.bc_values)
            )

        self.bc_dirichlet = ~self.bc_neumann
    
    def _set_boundary_size(self):
        self._lx = np.max([b._lx for b in self.boundaries])
        self._ly = np.max([b._ly for b in self.boundaries])

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

        for boundary in self.boundaries:
            if not boundary.is_on_domain_interior(point):
                # If it finds that the point is outside 
                # one of the boundaries, exit loop.
                return False

        return True
    
    def save(self, filename):
        """Save boundary data to file."""

        with open(filename, 'w') as file:
            # File header.
            number_of_boundaries = len(self.boundaries)
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
            for i, boundary in enumerate(self.boundaries):
                file.write(f'!boundary {i}\n')
                for j in range(boundary.number_of_elements):
                    file.write(
                        f'{boundary.vertices[j, 0]:.15e}    '
                        f'{boundary.vertices[j, 1]:.15e}    '
                        f'{boundary.bc_types[j]}    '
                        f'{boundary.bc_values[j]:.15e}\n'
                    )
                if i < number_of_boundaries - 1:
                    file.write(f'\n')

    def _load(self, filename):
        """Load boundary data from file."""

        def create_polygonal_boundary(vertices, bc_types, bc_values):
            # The last vertex must be equal to the first to form a closed boundary.
            vertices.append(vertices[0])

            # converting data to numpy arrays.
            vertices = np.array(vertices, np.float64)
            bc_types = np.array(bc_types, dtype=np.int8)
            bc_values = np.array(bc_values, dtype=np.float64)

            self.boundaries.append(Polygon(vertices, bc_types, bc_values))

        self.boundaries = []
        with open(filename, 'r') as file:
            i = 0
            for line in file:
                data_line = line.strip().lower()
                if data_line and data_line[0] != '#':
                    # Skip empty and comment lines.
                    if '!boundary' in data_line:
                        # Boundary data.
                        if i > 0:
                            create_polygonal_boundary(vertices, bc_types, bc_values)
                            i -= 1
                        i += 1
                        vertices = []
                        bc_types = []
                        bc_values = []
                        continue
                    else:
                        x, y, bc_type, bc_value = data_line.split()
                        vertices.append([x, y])
                        bc_types.append(bc_type)
                        bc_values.append(bc_value)

            create_polygonal_boundary(vertices, bc_types, bc_values)
            self.number_of_boundaries = len(self.boundaries)

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

        for i, boundary in enumerate(self.boundaries):
            if i == 0 and boundary.orientation < 0:
                ax.set_facecolor('silver')

            if boundary.orientation > 0:
                color = 'silver'
            else:
                color = 'white'
            
            bpatch = PolygonPatch(boundary.vertices, color=color)
            ax.add_patch(bpatch)
            ax.plot(
                boundary.vertices[:, 0],
                boundary.vertices[:, 1],
                'r-',
                marker='o',
                markersize=3,
                markerfacecolor='b',
                markeredgecolor='k',
                linewidth=2,
            )

        if show_element_index:        
            lref = np.min([self._lx, self._ly])
            index_offset = 0.03 * lref
            
            for i, element in enumerate(self.elements):
                index_position = element.node - index_offset * element.normal
                ax.text(
                    *index_position, i, ha='center', va='center', fontsize='x-small'
                )

        if filename:
            plt.savefig(filename, bbox_inches='tight')

        plt.show()

    def __neg__(self):
        B = Boundary()
        B.boundaries = [-b for b in self.boundaries]
        B._set_boundary_properties()

        return B

    def __sub__(self, other):
        return self.__add__(-other)

    def __add__(self, other):
        if isinstance(other, Polygon):
            boundaries = self.boundaries + [other]
        elif isinstance(other, Boundary):
            boundaries = self.boundaries + [b for b in other.boundaries]
        else:
            raise ValueError(f"Operand must be a Polygon or a Boundary")

        B = Boundary()
        B.boundaries = boundaries
        B._set_boundary_properties()

        return B

    @classmethod
    def rectangle(
        cls,
        bottom_left_corner,
        width,
        height,
        number_of_width_elements,
        number_of_height_elements,
        boundary_condition,
    ):
        """Create a rectangular boundary."""

        from twodubem.geometry import Rectangle

        R = Rectangle(
            bottom_left_corner,
            width,
            height,
            number_of_width_elements,
            number_of_height_elements,
            boundary_condition,
        )
        B = Boundary()
        B.boundaries.append(R)
        B._set_boundary_properties()

        return B

    @classmethod
    def square(
        cls,
        bottom_left_corner,
        side_length,
        number_of_side_elements,
        boundary_condition,
    ):
        """Create a square boundary."""

        from twodubem.geometry import Square

        S = Square(
            bottom_left_corner,
            side_length,
            number_of_side_elements,
            boundary_condition,
        )
        B = Boundary()
        B.boundaries.append(S)
        B._set_boundary_properties()

        return B

    @classmethod
    def circle(
        cls,
        center,
        radius,
        number_of_circumference_elements,
        boundary_condition,
        number_of_radius_elements=0,
        angle1=0.0,
        angle2=2*np.pi,
    ):
        """Create a circular boundary."""

        from twodubem.geometry import Circle

        C = Circle(
            center,
            radius,
            number_of_circumference_elements,
            boundary_condition,
            number_of_radius_elements,
            angle1,
            angle2,
        )
        B = Boundary()
        B.boundaries.append(C)
        B._set_boundary_properties()

        return B


#    ┏┓┳┓  ┳┓┏┓┳┳┓    Licensed under the MIT License
#    ┏┛┃┃┓┏┣┫┣ ┃┃┃    Copyright (c) 2025 Rodrigo Castro
#    ┗━┻┛┗┻┻┛┗┛┛ ┗    https://github.com/rodpcastro/twodubem

"""
Solver
======

This module contains the classes to solve Laplace's equation in two dimensions.

Classes
-------
Solver
    Solution of 2D Laplace's equation using the Boundary Element Method.
"""

import numpy as np
from twodubem._internal import tdb_warn

class Solver:
    """Solution of a boundary value problem using the Boundary Element Method.

    Parameters
    ----------
    boundary : Boundary
        Boundary of the region where the differential operator acts.
    green_function : Green
        Green's function associated with the differential operator.

    Attributes
    ----------
    boundary : Boundary
        Boundary of the region where the differential operator acts.
    green : Green
        Green's function associated with the differential operator.

    Methods
    -------
    show_influence_matrices()
        Display a graphical representation of the influence matrices.
    solve()
        Solve the boundary value problem.
    get_solution(X, Y)
        Get solution at a given grid of points.
    """

    def __init__(self, boundary, green_function):
        self.boundary = boundary
        self.green = green_function

    def _build_influence_matrices(self):
        """Build influence coefficients matrices."""
 
        n = self.boundary.number_of_elements
        self.G = np.empty((n, n), dtype=np.float64)
        self.Q = np.empty((n, n), dtype=np.float64)

        for i, source_element in enumerate(self.boundary.elements):
            source_point = source_element.node
            for j, field_element in enumerate(self.boundary.elements):
                g, q, _, _ = self.green.get_line_element_influence_coefficients(field_element, source_point, show_warnings=False)
                self.G[i, j] = g
                self.Q[i, j] = q
                if i == j:
                    self.Q[i, j] += -0.5

    def show_influence_matrices(self):
        """Display a graphical representation of the influence matrices."""

        import matplotlib.pyplot as plt

        if not hasattr(self, 'G') or not hasattr(self, 'Q'):
            self._build_influence_matrices()
        
        fig, ax = plt.subplots(1, 2, figsize=(7,4))
        
        matG = ax[0].matshow(self.G)
        ax[0].set_title(r'$\mathbb{G}$')
        
        matQ = ax[1].matshow(self.Q)
        ax[1].set_title(r'$\mathbb{Q}$')

        Gmin = self.G.min()
        Gmax = self.G.max()
        Gavg = 0.5 * (Gmin + Gmax)
        Gticks = [Gmin, Gavg, Gmax]

        Qmin = self.Q.min()
        Qmax = self.Q.max()
        Qavg = 0.5 * (Qmin + Qmax)
        Qticks = [Qmin, Qavg, Qmax]
        
        fig.colorbar(
            matG,
            ax=ax[0],
            location='bottom',
            shrink=0.9,
            format='%.1e',
            pad=0.05,
            ticks=Gticks,
        )
        fig.colorbar(
            matQ,
            ax=ax[1],
            location='bottom',
            shrink=0.9,
            format='%.1e',
            pad=0.05,
            ticks=Qticks,
        )
        
        plt.show()

    def solve(self):
        """Solve the boundary integral system of equations."""

        self._build_influence_matrices()
        
        n = self.boundary.number_of_elements
        A = np.empty((n, n), dtype=np.float64)
        C = np.empty((n, n), dtype=np.float64)
        u = np.zeros(n, dtype=np.float64)
        q = np.zeros(n, dtype=np.float64)

        bc_neumann = self.boundary.bc_neumann
        bc_dirichlet = self.boundary.bc_dirichlet

        A[:, bc_dirichlet] = -self.G[:, bc_dirichlet]
        A[:, bc_neumann] = self.Q[:, bc_neumann]

        C[:, bc_dirichlet] = -self.Q[:, bc_dirichlet]
        C[:, bc_neumann] = self.G[:, bc_neumann]
        
        b = C @ self.boundary.bc_values_

        x = np.linalg.solve(A, b)

        u[bc_dirichlet] = self.boundary.bc_values_[bc_dirichlet]
        u[bc_neumann] = x[bc_neumann]

        q[bc_neumann] = self.boundary.bc_values_[bc_neumann]
        q[bc_dirichlet] = x[bc_dirichlet]

        self.u = u
        self.q = q

    def get_solution(self, X, Y, check_points=True, show_warnings=False):
        """Get solution for array of points.

        Parameters
        ----------
        X : ndarray[float], shape=(n, m)
            Array with points' x-coordinates.
        Y : ndarray[float], shape=(n, m)
            Array with points' y-coordinates.
        check_points : bool, default=True
            Check if point is on the boundary, in the domain's interior or outside the domain.
        show_warnings : bool, default=False
            Show warning messages.

        Returns
        -------
        Z : ndarray[float], shape=(n, m)
            Solution at the array of points.
        W : ndarray[float], shape=(n, m, 2)
            Gradient of the solution at the array of points.
        """

        n = self.boundary.number_of_elements
        x = X.ravel()
        y = Y.ravel()
        z = np.empty(x.shape)
        w = np.empty((*x.shape, 2))
        G = np.empty(n)
        Q = np.empty(n)
        gradG = np.empty((2, n))
        gradQ = np.empty((2, n))
        for i in range(len(x)):

            field_point = np.array([x[i], y[i]])

            if check_points:
                on_boundary, element_index = self.boundary.is_on_boundary(field_point)
                if on_boundary:
                    z[i] = self.u[element_index]  # TODO: edit this once the linear element is implemented.
                    w[i] = np.nan
                    if show_warnings:
                        tdb_warn(f"Point ({x[i]}, {y[i]}) is on the boundary. Returning NaN for the gradient.")
                    continue
                elif not self.boundary.is_on_domain_interior(field_point):
                    z[i] = np.nan
                    w[i] = np.nan
                    if show_warnings:
                        tdb_warn(f"Point ({x[i]}, {y[i]}) is outside the domain. Returning NaN instead.")
                    continue

            for j, source_element in enumerate(self.boundary.elements):
                
                g, q, gradg, gradq = self.green.get_line_element_influence_coefficients(source_element, field_point, show_warnings=show_warnings)
                
                G[j] = g
                Q[j] = q
                gradG[:, j] = gradg
                gradQ[:, j] = gradq

            z[i] = Q @ self.u - G @ self.q
            w[i] = gradQ @ self.u - gradG @ self.q

        Z = z.reshape(X.shape)
        W = w.reshape((*X.shape, 2))

        return Z, W


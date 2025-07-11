#    ┏┓┳┓  ┳┓┏┓┳┳┓    Licensed under the MIT License
#    ┏┛┃┃┓┏┣┫┣ ┃┃┃    Copyright (c) 2025 Rodrigo Castro
#    ┗━┻┛┗┻┻┛┗┛┛ ┗    https://github.com/rodpcastro/twodubem

"""
Solver
======

This module contains the class to solve boundary value problems.

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
    method : str, default='constant'
        Approximation method used for the solution function. Available methods are
        'constant' and 'linear'.

    Attributes
    ----------
    boundary : Boundary
        Boundary of the region where the differential operator acts.
    green : Green
        Green's function associated with the differential operator.
    method : str
        Approximation method used for the solution function.
    u : ndarray[float], shape=(n, 2)
        Solution at the nodes.
    q : ndarray[float], shape=(n, 2)
        Normal derivative at the nodes.
    r : ndarray[float], shape=(n, 2)
        Tangential derivative at the nodes.

    Methods
    -------
    show_influence_matrices()
        Display a graphical representation of the influence matrices.
    solve()
        Solve the boundary value problem.
    show_boundary_solution(filename='')
        Display graphical representation of the solution on the boundary.
    get_solution(X, Y)
        Get solution at a given grid of points.
    """

    def __init__(self, boundary, green_function, method='constant'):
        self.boundary = boundary
        self.green = green_function
        self.method = method

    def _build_influence_matrices(self):
        """Build influence coefficients matrices."""
 
        n = self.boundary.number_of_elements
        self.G = np.empty((n, n), dtype=np.float64)
        self.Q = np.empty((n, n), dtype=np.float64)

        for i, source_element in enumerate(self.boundary.elements):
            source_point = source_element.node
            for j, field_element in enumerate(self.boundary.elements):
                g, q, _, _ = self.green.get_line_element_influence_coefficients(
                        field_element,
                        source_point,
                        method=self.method,
                        show_warnings=False,
                )
                self.G[i, j] = g
                self.Q[i, j] = q
                if i == j:
                    self.Q[i, j] += -0.5

    def show_influence_matrices(self, filename=''):
        """Display a graphical representation of the influence matrices.

        Paramaters
        ----------
        filename : str, default=''
            Name of the file in which the figure is saved. If not specified, the figure
            is not saved to a file.
        """

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
        
        if filename:
            plt.savefig(filename, bbox_inches='tight')

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
        
        b = C @ self.boundary.bc_values

        x = np.linalg.solve(A, b)

        u[bc_dirichlet] = self.boundary.bc_values[bc_dirichlet]
        u[bc_neumann] = x[bc_neumann]

        q[bc_neumann] = self.boundary.bc_values[bc_neumann]
        q[bc_dirichlet] = x[bc_dirichlet]

        self.u = u  # Solution at the nodes.
        self.q = q  # Normal derivative at the nodes.
        
        self._get_tangential_derivative_on_boundary()

    def _get_tangential_derivative_on_boundary(self):
        # Allowing the evaluation of the gradient on the boundary. By default, TwoDuBEM
        # returns NaN gradients on the boundary because this option is set to False.
        self.green._eval_on_boundary = True

        xm = self.boundary.midpoints[:, 0]
        ym = self.boundary.midpoints[:, 1]

        _, wm = self.get_solution(xm, ym)

        # Multiplying by 2.0, because the gradient w is evaluated on the boundary.
        r = 2.0 * np.sum(wm * self.boundary.tangents, axis=1)

        # Back to TwoDuBEM's default configuration.
        self.green._eval_on_boundary = False

        self.r = r  # Tangential derivative at the nodes.

    def show_boundary_solution(self, filename=''):
        """Display graphical representation of the solution on the boundary.

        Paramaters
        ----------
        filename : str, default=''
            Name of the file in which the figure is saved. If not specified, the figure
            is not saved to a file.
        """

        import matplotlib.pyplot as plt
        from matplotlib import ticker

        fig, ax = plt.subplots(figsize=(5,4))

        n_start = 0
        for i, boundary in enumerate(self.boundary.boundaries):
            n = boundary.number_of_elements
            n_end = n_start + n
            v = np.empty(2*n, dtype=np.int16)
            z = np.empty(2*n, dtype=np.float64)
            for j in range(n_start, n_end):
                v[2*j:2*(j+1)] = [j, j+1]
                if self.method == 'constant':
                    z[2*j:2*(j+1)] = self.u[j]
                elif self.method == 'linear':
                    z[2*j:2*(j+1)] = self.u[j:j+1]
    
            ax.plot(v, z, label=rf'$B_{i}$')
            n_start = n_end

        ax.set_title('Solution on the boundary')
        ax.set_xlabel('Vertex index')
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
        plt.legend(loc='upper right')
        plt.grid()

        if filename:
            plt.savefig(filename, bbox_inches='tight')

        plt.show()
    
    def _get_boundary_solution(self, element_index, point):
        """Get solution for point on the boundary."""

        if self.method == 'constant':
            return self.u[element_index]
        elif self.method == 'linear':
            raise NotImplementedError

    def get_solution(self, X, Y, check_points=False, show_warnings=False):
        """Get solution for array of points.

        Parameters
        ----------
        X : ndarray[float], shape=(n, m)
            Array with points' x-coordinates.
        Y : ndarray[float], shape=(n, m)
            Array with points' y-coordinates.
        check_points : bool, default=False
            Check if point is on the boundary, in the domain's interior or outside the
            domain. If set to ``True``, computations are slower. If it's known that all
            points are in the domain's interior, keep this option ``False``.
        show_warnings : bool, default=False
            Show warning messages.

        Returns
        -------
        Z : ndarray[float], shape=(n, m)
            Solution at the array of points.
        W : ndarray[float], shape=(n, m, 2)
            Gradient of the solution at the array of points.

        Warns
        -----
        TDBWarning
            Returns NaN for points outside the domain, and returns NaN gradients for
            points on the boundary.
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
                    z[i] = self._get_boundary_solution(element_index, point)
                    w[i] = np.nan
                    if show_warnings:
                        tdb_warn(f"Point ({x[i]}, {y[i]}) is on the boundary. "
                                 f"Returning NaN for the gradient.")
                    continue
                elif not self.boundary.is_on_domain_interior(field_point):
                    z[i] = np.nan
                    w[i] = np.nan
                    if show_warnings:
                        tdb_warn(f"Point ({x[i]}, {y[i]}) is outside the domain. "
                                 f"Returning NaN instead.")
                    continue

            for j, source_element in enumerate(self.boundary.elements):

                Gj, Qj, gGj, gQj = self.green.get_line_element_influence_coefficients(
                    source_element,
                    field_point,
                    method=self.method,
                    show_warnings=show_warnings,
                )
                G[j] = Gj
                Q[j] = Qj
                gradG[:, j] = gGj
                gradQ[:, j] = gQj

            z[i] = Q @ self.u - G @ self.q
            w[i] = gradQ @ self.u - gradG @ self.q

        Z = z.reshape(X.shape)
        W = w.reshape((*X.shape, 2))

        return Z, W


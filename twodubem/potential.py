"""
Potential
=========

This module contains the classes to solve Laplace's equation in two dimensions.

Classes
-------
BEMPotential
    Solution of 2D Laplace's equation using the Boundary Element Method.
"""

import numpy as np

class BEMPotential:
    # TODO: Edit get_solution to also obtain solution for boundary points.

    """Solution of 2D Laplace's equation using the Boundary Element Method.

    Parameters
    ----------
    boundary : Boundary
        Boundary of the region where the Laplace's operator act.

    Attributes
    ----------
    boundary : Boundary
        Boundary of the region where the Laplace's operator act.

    Methods
    -------
    build_influence_matrices()
        Build matrices containing the influence coefficients.
    show_influence_matrices()
        Display a graphical representation of the influence matrices.
    solve()
        Solve the boundary integral system of equations.
    get_interior_solution(X, Y)
        Get solution at a given grid of points.
    """

    def __init__(self, boundary):
        self.boundary = boundary

    def build_influence_matrices(self):
        """Build matrices containing the influence coefficients."""
 
        n = self.boundary.number_of_elements
        self.G = np.empty((n, n), dtype=np.float64)
        self.Q = np.empty((n, n), dtype=np.float64)
        
        for i, field_element in enumerate(self.boundary.elements):
            for j, source_element in enumerate(self.boundary.elements):                
                g, q = source_element.get_influence_coefficients(field_element.node)
                self.G[i, j] = g
                self.Q[i, j] = q
                if i == j:
                    self.Q[i, j] += -0.5
    
    def show_influence_matrices(self):
        """Display a graphical representation of the influence matrices."""

        import matplotlib.pyplot as plt
        
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

        self.build_influence_matrices()
        
        n = self.boundary.number_of_elements
        A = np.empty((n, n), dtype=np.float64)
        C = np.empty((n, n), dtype=np.float64)
        u = np.zeros(n, dtype=np.float64)
        q = np.zeros(n, dtype=np.float64)

        bc_neumann = self.boundary.bc_types.astype(np.bool)
        bc_dirichlet = ~bc_neumann

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

        self.u = u
        self.q = q

    def get_interior_solution(self, X, Y, return_gradient=False):
        """Get solution at a given grid of points.

        Parameters
        ----------
        X : ndarray[float], shape=(n,m)
            Array with points' x-coordinates.
        Y : ndarray[float], shape=(n,m)
            Array with points' y-coordinates.
        return_gradient : bool, default=False
            If ``True``, the solution gradient is returned.

        Returns
        -------
        Z : ndarray[float], shape=(n,m)
            Solution at the given grid of points.
        W : ndarray[float], shape=(n,m,2)
            Gradient of the solution at the given grid of points. Returned when
            ``return_gradient=True``.
        """

        x = X.ravel()
        y = Y.ravel()
        n = self.boundary.number_of_elements
        G = np.empty((len(x), n))
        Q = np.empty((len(x), n))

        if return_gradient:
            gradG = np.empty((len(x), 2, n))
            gradQ = np.empty((len(x), 2, n))

        for i in range(len(x)):
            for j, boundary_element in enumerate(self.boundary.elements):
                point = np.array([x[i], y[i]])
                
                if return_gradient:
                    g, q, gradg, gradq = boundary_element.get_influence_coefficients(
                        point,
                        return_gradients=True,
                    )
                    gradG[i, :, j] = gradg
                    gradQ[i, :, j] = gradq
                else:
                    g, q = boundary_element.get_influence_coefficients(point)

                G[i, j] = g
                Q[i, j] = q

        z = Q @ self.u - G @ self.q
        Z = z.reshape(X.shape)

        if return_gradient:
            w = gradQ @ self.u - gradG @ self.q
            W = w.reshape((*X.shape, 2))

            return Z, W

        return Z

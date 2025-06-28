#    ┏┓┳┓  ┳┓┏┓┳┳┓    Licensed under the MIT License
#    ┏┛┃┃┓┏┣┫┣ ┃┃┃    Copyright (c) 2025 Rodrigo Castro
#    ┗━┻┛┗┻┻┛┗┛┛ ┗    https://github.com/rodpcastro/twodubem

"""
Laplace
=======

This module contains the class for the Green's function of 2D Laplace's equation.

Classes
-------
Laplace
    Green's function of 2D Laplace's equation.
"""

import numpy as np
from twodubem.green import Green
from twodubem._internal import ismall


class Laplace(Green):
    """Green's function of 2D Laplace's equation."""

    def get_constant_element_influence_coefficients(
        self, element, point, return_gradients=False
    ):
        """Get influence coefficients of a constant element at a field point.

        Parameters
        ----------
        element : StraightConstantElement
            Straight constant boundary element.
        point : ndarray, shape=(2,)
            Field point's coordinates in global system.
        return_gradients : bool, default=False
            If ``True``, The gradients of G and Q in the global coordinate system are
            computed and returned.

        Returns
        -------
        G : float
            Integral of the Green's function over the element.
        Q : float
            Integral of the Green's function normal derivative over the element.
        gradG : ndarray[float], shape=(2,)
            Gradient of G in the global coordinate system.
        gradQ : ndarray[float], shape=(2,)
            Gradient of Q in the global coordinate system.

        Warns
        -----
        TDBWarning
            Singular behavior for the gradients of G and Q at a point too close or 
            coincident with the element's edges.
        TDBWarning
            Inaccuracy at the computation of the gradients of G and Q at a point too
            close or inside the element.
        """

        x, y = element.get_point_local_coordinates(point)

        elength = element.length
        a = 0.5 * elength

        # Element interior (|x| < a, y = 0).
        is_element_interior = ismall(y, elength) and np.abs(x) < a

        # Element's edges (|x| = a, y = 0).
        is_element_edge = ismall(y, elength) and ismall(np.abs(x) - a, elength)
       
        # Q is discontinuous for (|x| ≤ a, y = 0). Returns Q = 0.0 in this region.
        if is_element_edge:
            G = a / np.pi * (np.log(2*a) - 1.0)
            Q = 0.0
        else:
            hp = 0.5 / np.pi
            
            xpa = x + a
            xma = x - a

            r1 = np.sqrt(xma**2 + y**2)
            r2 = np.sqrt(xpa**2 + y**2)
            t1 = np.arctan2(y, xma)
            t2 = np.arctan2(y, xpa)

            if is_element_interior:
                # Not sure if this is necessary. Only used for the gradients.
                t1m2 = -np.pi
            else:
                t1m2 = t1 - t2

            G = hp * (y * t1m2 - xma * np.log(r1) + xpa * np.log(r2) - 2.0 * a)

            if ismall(y, elength):
                Q = 0.0
            else:
                Q = -hp * t1m2
            
        if not return_gradients:
            return G, Q
        
        if is_element_edge:
            gradG = np.array([np.nan, np.nan], dtype=np.float64)
            gradQ = np.array([np.nan, np.nan], dtype=np.float64)
            tdb_warn("Gradients of G and Q are singular for a point too close or "
                     "coincident with the element's edges. Returning NaN instead.")
        else:
            if is_element_interior:
                tdb_warn("Gradients of G and Q are inaccurate for "
                         "a point too close or inside the element.")

            cosb = self.tangent[0]
            sinb = self.tangent[1]

            Gx1 = np.log(r1 / r2)
            Gx2 = t1m2
            
            Gx = -hp * (Gx1 * cosb - Gx2 * sinb)
            Gy = -hp * (Gx1 * sinb + Gx2 * cosb)

            Qx1 = y / r1**2 - y / r2**2
            Qx2 = xma / r1**2 - xpa / r2**2
            
            Qx = hp * (Qx1 * cosb - Qx2 * sinb)
            Qy = hp * (Qx1 * sinb + Qx2 * cosb)

            gradG = np.array([Gx, Gy], dtype=np.float64)
            gradQ = np.array([Qx, Qy], dtype=np.float64)

        return G, Q, gradG, gradQ


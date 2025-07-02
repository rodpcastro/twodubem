#    ┏┓┳┓  ┳┓┏┓┳┳┓    Licensed under the MIT License
#    ┏┛┃┃┓┏┣┫┣ ┃┃┃    Copyright (c) 2025 Rodrigo Castro
#    ┗━┻┛┗┻┻┛┗┛┛ ┗    https://github.com/rodpcastro/twodubem

"""
Laplace
=======

This module contains the Green's function class for 2D Laplace's equation.

Classes
-------
Laplace
    Green's function for 2D Laplace's equation.

References
----------
1. Steven L. Crouch, Sofia G. Mogilevskaya. 2024. A First Course in
   Boundary Element Methods. Springer, Cham, Switzerland.
"""

import numpy as np
from twodubem.green import Green
from twodubem._internal import ismall, tdb_warn


class Laplace(Green):
    """Green's function for 2D Laplace's equation."""

    @staticmethod
    def eval(field_point, source_point):
        """Green's function evaluation."""

        fx, fy = field_point
        sx, sy = source_point

        ip = 1.0 / np.pi
        hp = 0.5 * ip
        dx = fx - sx
        dy = fy - sy
        dx2 = dx**2
        dy2 = dy**2
        ds = dx2 + dy2
        dq = ds**2
        
        # Green's function.
        g = 0.5 * hp * np.log(ds)

        # Green's function gradient.
        gx = hp * dx / ds
        gy = hp * dy / ds
        gradg = np.array([gx, gy])

        # Green's function gradient's gradient. 
        gxx = -hp * (dx2 - dy2) / dq
        gxy = -ip * dx * dy / dq
        gyy = -gxx
        gyx = gxy

        gradk = np.array([[gxx, gyx],
                          [gxy, gyy]])

        return g, gradg, gradk

    def _get_line_element_constant_influence_coefficients(
        self, field_element, source_point, show_warnings=True,
    ):
        """Get constant influence coefficients on a line element.

        Parameters
        ----------
        field_element : LineElement
            Field element, where the integration is carried over.
        source_point : ndarray[float], shape=(2,)
            Source point, where the source is located.
        show_warnings : bool, default=True
            Show warning messages.

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
            Inaccurate or singular behavior for the influence coefficients gradients 
            when the source point is too close or on a boundary element.
        """

        x, y = field_element.get_point_local_coordinates(source_point)

        elength = field_element.length
        a = 0.5 * elength

        # Source point on field element (|x| ≤ a, |y| = 0).
        is_source_on_element = ismall(y, elength) and np.abs(x) <= a

        # Source point on field element's endpoints (|x| = a, |y| = 0).
        is_source_on_endpoint = ismall(y, elength) and ismall(np.abs(x) - a, elength)
       
        # Source point on field element's node (|x| = 0, |y| = 0).
        is_source_on_node = ismall(x, elength) and ismall(y, elength)

        # Q is discontinuous for (|x| ≤ a, y = 0). Returns Q = 0.0 in this region.
        if is_source_on_endpoint:
            G = a / np.pi * (np.log(2.0 * a) - 1.0)
            Q = 0.0
        elif is_source_on_node:
            G = a / np.pi * (np.log(a) - 1.0)
            Q = 0.0
        else:
            hp = 0.5 / np.pi
            
            xpa = x + a
            xma = x - a

            r1 = np.sqrt(xma**2 + y**2)
            r2 = np.sqrt(xpa**2 + y**2)
            t1 = np.arctan2(y, xma)
            t2 = np.arctan2(y, xpa)
            t1m2 = t1 - t2

            G = hp * (y * t1m2 - xma * np.log(r1) + xpa * np.log(r2) - 2.0 * a)

            if ismall(y, elength):
                Q = 0.0
            else:
                Q = -hp * t1m2

        if is_source_on_element:
            gradG = np.array([np.nan, np.nan], dtype=np.float64)
            gradQ = np.array([np.nan, np.nan], dtype=np.float64)
            if show_warnings:
                tdb_warn("Influence coefficients gradients are inaccurate "
                         "or singular for a source point too close or on "
                         "a boundary element. Returning NaN instead.")
        else:
            # I had to add a negative sign in the definition of the jacobian matrix to
            # get the correct signs for the gradients. I'm not sure, but I believe the
            # reason is the fact that my local coordinate system is left handed, while
            # the reference (Crouch, 2024) uses a right handed local system.
            J = -np.vstack((field_element.tangent, field_element.normal)).T

            Gx = -np.log(r1 / r2)
            Gy = t1m2
            gradG = hp * J @ np.array([Gx, Gy])

            Qx = -y / r1**2 + y / r2**2
            Qy = xma / r1**2 - xpa / r2**2
            gradQ = -hp * J @ np.array([Qx, Qy])
            
        return G, Q, gradG, gradQ


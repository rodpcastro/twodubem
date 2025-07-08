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
        self, element, point, show_warnings=True,
    ):
        """Get constant influence coefficients on a line element.

        Warns
        -----
        TDBWarning
            Inaccurate or singular behavior for the influence coefficients' gradients 
            when the point is too close or on the boundary.
        """

        x, y = element.get_point_local_coordinates(point)

        elength = element.length
        a = 0.5 * elength

        # Point on element (|x| ≤ a, |y| = 0).
        is_point_on_element = ismall(y, elength) and np.abs(x) <= a

        # Point on element's endpoints (|x| = a, |y| = 0).
        is_point_on_endpoint = ismall(y, elength) and ismall(np.abs(x) - a, elength)

        # Point on element's node (|x| = 0, |y| = 0).
        is_point_on_node = ismall(x, elength) and ismall(y, elength)

        # Q is discontinuous for (|x| ≤ a, |y| = 0). Returns Q = 0.0 in this region.
        if is_point_on_endpoint:
            G = a / np.pi * (np.log(2.0 * a) - 1.0)
            Q = 0.0
        elif is_point_on_node:
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

        # Gradients are calculated for a point on the domain's interior. They are not
        # used to obtain the solution at the boundary. On the other hand, Q, the normal
        # derivative of G, is calculated on the boundary. This explains why the dot
        # product between gradG and the element normal vector is -Q.
        if is_point_on_element:
            gradG = np.array([np.nan, np.nan], dtype=np.float64)
            gradQ = np.array([np.nan, np.nan], dtype=np.float64)
            if show_warnings:
                tdb_warn("Influence coefficients' gradients are inaccurate or "
                         "singular for a point too close or on the boundary. "
                         "Returning NaN instead.")
        else:
            # Jacobian matrix.
            J = np.vstack((element.tangent, element.normal)).T

            Gx = -np.log(r1 / r2)
            Gy = t1m2
            gradG = hp * J @ np.array([Gx, Gy])

            Qx = -y / r1**2 + y / r2**2
            Qy = xma / r1**2 - xpa / r2**2
            gradQ = -hp * J @ np.array([Qx, Qy])

        return G, Q, gradG, gradQ


#    ┏┓┳┓  ┳┓┏┓┳┳┓    Licensed under the MIT License
#    ┏┛┃┃┓┏┣┫┣ ┃┃┃    Copyright (c) 2025 Rodrigo Castro
#    ┗━┻┛┗┻┻┛┗┛┛ ┗    https://github.com/rodpcastro/twodubem

"""
Element
=======

This module contains classes to create boundary elements.

Classes
-------
StraightConstantElement
    Straight constant boundary element.
StraightLinearElement
    Straight linear boundary element. (not implemented)
"""

import numpy as np
from numpy import ndarray
from twodubem._internal import ismall, tdb_warn


class Element:
    # TODO: Implement parent class. This one will also be used for type annotations.
    pass


class StraightConstantElement(Element):
    """Straight constant boundary element.

    Parameters
    ----------
    point1 : ndarray[float], shape=(2,)
        First end point global coordinates.
    point2 : ndarray[float], shape=(2,)
        Second end point global coordinates.
        
    Attributes
    ----------
    endpoints : ndarray[float], shape=(2,2)
        Element end points' global coordinates.
    node : ndarray[float], shape=(2,)
        Midpoint between endpoints.
    length : float
        Element's length.
    tangent : ndarray[float], shape=(2,)
        Element's tangent vector.
    normal : ndarray[float], shape=(2,)
        Element's normal vector.

    Methods
    -------
    get_point_local_coordinates(point)
        Get point's coordinates in local system.
    get_point_distance(point)
        Get point's distance from the element.
    get_influence_coefficients(point, return_gradients=False)
        Get influence coefficients at a point.
    """

    def __init__(self, point1, point2):
        self.endpoints = np.array([point1, point2])
        self.node = self.endpoints.mean(axis=0)

        r = self.endpoints[1] - self.endpoints[0]
        self.length = np.linalg.norm(r)
        self.tangent = r / self.length
        self.normal = np.array([self.tangent[1], -self.tangent[0]])
        
    def get_point_local_coordinates(self, point):
        """Get point's coordinates in the local system.

        Parameters
        ----------
        point : ndarray, shape=(2,)
            Point's coordinates in the global system.

        Returns
        -------
        point_local : ndarray, shape=(2,)
            Point's coordinates in the local system.
        """

        dif = np.array(point) - self.node
        x = np.dot(dif, self.tangent)
        y = np.dot(dif, self.normal)
        point_local = np.array([x, y])

        return point_local

    def get_point_distance(self, point):
        """Get point's distance from the element.
        
        Parameters
        ----------
        point : ndarray, shape=(2,)
            Point coordinates in the global system.

        Returns
        -------
        distance : float
            Point's distance from the element.
        """

        a = 0.5 * self.length
        x, y = self.get_point_local_coordinates(point)

        if np.abs(x) <= a:
            distance = np.abs(y)
        else:
            dif_from_endpoints = point - self.endpoints
            distance_from_endpoints = np.linalg.norm(dif_from_endpoints, axis=1)
            distance = distance_from_endpoints.min()

        return distance
    
    def get_influence_coefficients(self, field_global, return_gradients=False):
        """Get influence coefficients at a field point.

        Parameters
        ----------
        field_global : ndarray, shape=(2,)
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

        x, y = self.get_point_local_coordinates(field_global)

        a = 0.5 * self.length

        # Element interior (|x| < a, y = 0).
        is_element_interior = ismall(y, self.length) and np.abs(x) < a

        # Element's edges (|x| = a, y = 0).
        is_element_edge = ismall(y, self.length) and ismall(np.abs(x) - a, self.length)
       
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
                # FIXME: Not sure if this is necessary.
                t1m2 = -np.pi
            else:
                t1m2 = t1 - t2

            G = hp * (y * t1m2 - xma * np.log(r1) + xpa * np.log(r2) - 2.0 * a)

            if ismall(y, self.length):
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


class StraightLinearElement(StraightConstantElement):
    # TODO: Implement.
    pass

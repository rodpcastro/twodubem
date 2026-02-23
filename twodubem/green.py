#    ┏┓┳┓  ┳┓┏┓┳┳┓    Licensed under the MIT License
#    ┏┛┃┃┓┏┣┫┣ ┃┃┃    Copyright (c) 2025 Rodrigo Castro
#    ┗━┻┛┗┻┻┛┗┛┛ ┗    https://github.com/rodpcastro/twodubem

"""
Green
=====

This module contains the base class for Green's functions.

Classes
-------
Green
    Base class for all Green's functions.
"""

import numpy as np


class Green:

    @staticmethod
    def eval(field_point, source_point, *args, **kwargs):
        """Green's function evaluation."""

        raise NotImplementedError

    def get_line_element_influence_coefficients(
        self, element, point, method='constant', *args, **kwargs,
    ):
        """Get influence coefficients on a line element.

        Parameters
        ----------
        element : LineElement
            Element where the integration is carried over.
        point : ndarray[float], shape=(2,)
            Source point for the evaluation of the influence coefficients, and a field 
            point for the evaluation of its gradients.
        method : str, default='constant'
            Approximation method used for the solution function. Available methods are
            'constant' and 'linear'.
        show_warnings : bool, default=True
            Show warning messages.

        Returns
        -------
        G : float
            Integral of the Green's function over the element.
        Q : float
            Integral of the Green's function normal derivative over the element.
        gradG : ndarray[float], shape=(2,)
            Gradient of G in the global coordinate system evaluated at the point.
        gradQ : ndarray[float], shape=(2,)
            Gradient of Q in the global coordinate system evaluated at the point.

        Raises
        ------
        ValueError
            If solution approximation `method` is invalid.
        """

        method_ = method.strip().lower()

        if method_ == 'constant':
            return self._get_line_element_constant_influence_coefficients(
                element, point, *args, **kwargs,
            )
        elif method_ == 'linear':
            return self._get_line_element_linear_influence_coefficients(
                element, point, *args, **kwargs,
            )
        else:
            raise ValueError(f"Invalid solution approximation method '{method}'")

    def _get_line_element_constant_influence_coefficients(
        self, element, point, *args, **kwargs,
    ):
        """Get constant influence coefficients on a line element."""

        a = 0.5 * element.length

        # 4-point Gauss-Legendre quadrature roots and weights.
        roots = [0.3399810435848563, 0.8611363115940526]
        weights = [0.6521451548625461, 0.3478548451374538]
        
        G = 0.0
        gradG = np.zeros(2, dtype=np.float64)
        gradK = np.zeros((2, 2), dtype=np.float64)
        for i in range(len(roots)):
            element_point_p = element.get_point_global_coordinates(
                np.array([a * roots[i], 0.0])
            )
            element_point_m = element.get_point_global_coordinates(
                np.array([-a * roots[i], 0.0])
            )

            g_p, gradg_p, gradk_p = self.eval(element_point_p, point, *args, **kwargs)
            g_m, gradg_m, gradk_m = self.eval(element_point_m, point, *args, **kwargs)

            G += weights[i] * (g_p + g_m)
            gradG += weights[i] * (gradg_p + gradg_m)
            gradK += weights[i] * (gradk_p + gradk_m)

        G = a * G
        Q = a * gradG @ element.normal

        # The negative sign is used for the the gradients because they are calculated
        # for a point on the domain's interior. Q (the normal derivative of G), on the
        # other hand, is computed for a point on the boundary. The gradients are not
        # used to obtain the solution at the boundary.
        gradG = -a * gradG
        gradQ = -a * gradK @ element.normal

        return G, Q, gradG, gradQ

    def _get_line_element_linear_influence_coefficients(
        self, element, point, *args, **kwargs,
    ):
        """Get linear influence coefficients on a line element."""

        raise NotImplementedError


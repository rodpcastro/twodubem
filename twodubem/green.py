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


class Green:

    @staticmethod
    def eval(field, source):
        """Green's function evaluation."""
        raise NotImplementedError

    def get_constant_element_influence_coefficients(
        self, field_element, source_element, return_gradients=False
    ):
        """Get influence coefficients for a constant element."""

        source_point = source_element.node
        a = 0.5 * field_element.length

        # 4-point Gauss-Legendre quadrature roots and weights.
        roots = [0.3399810435848563, 0.8611363115940526]
        weights = [0.6521451548625461, 0.3478548451374538]
        
        G = 0.0
        gradG = np.zeros(2, dtype=np.float64)
        gradK = np.zeros((2, 2), dtype=np.float64)
        for i in range(len(roots)):
            field_point_p = field_element.get_point_global_coordinates(
                np.array([a * roots[i], 0.0])
            )
            field_point_m = field_element.get_point_global_coordinates(
                np.array([-a * roots[i], 0.0])
            )

            g_p, gradg_p, gradk_p = self.eval(field_point_p, source_point)
            g_m, gradg_m, gradk_m = self.eval(field_point_m, source_point)

            G += weights[i] * (g_p + g_m)
            gradG += weights[i] * (gradg_p + gradg_m)
            gradK += weights[i] * (gradk_p + gradk_m)

        G = a * G
        gradG = a * gradG
        Q = gradG @ field_element.normal
        gradQ = a * gradK @ field_element.normal

        return G, Q, gradG, gradQ

    def get_linear_element_influence_coefficients(
        self, field_element, source_element, return_gradients=False
    ):
        """Get influence coefficients for a linear element."""
        raise NotImplementedError


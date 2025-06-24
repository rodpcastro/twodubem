"""
Internal
========

This module contains variables and functions for internal use.

Variables
---------
eps : float
    Machine epsilon.

Functions
---------
ismall(x, reference=1.0)
    Evaluate the smallness of a variable compared to a reference value.

Classes
-------
TDBWarning
    Custom TDB warning.
"""

import numpy as np
import warnings


eps = np.finfo(np.float64).eps


def ismall(x, reference = 1.0):
    """Evaluate the smallness of a variable compared to a reference value."""

    return np.abs(x) < reference * eps


class TDBWarning(Warning):
    """Custom TDB warning.

    Issued anytime TDB finds an inconsistency in the user's input,
    but one that can be handled without halting the execution.
    """

    pass


def tdb_formatwarning(message, category, filename, lineno, line=None):
    return f'\033[93m{category.__name__}\033[0m: {message}\n'


def tdb_warn(message: str):
    """Displays a TDBWarning message with custom formattting.

    Parameters
    ----------
    message : str
        The message to be displayed in the warning.
    """

    original_formatwarning = warnings.formatwarning
    warnings.formatwarning = tdb_formatwarning
    warnings.warn(message, NodimoWarning)
    warnings.formatwarning = original_formatwarning

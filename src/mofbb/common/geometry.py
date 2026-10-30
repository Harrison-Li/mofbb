'''
Shared connection geometry operations.
'''

import numpy as np


def normalize_directions(positions: np.ndarray) -> np.ndarray:
    '''
    Center connection points and turn each centered vector into a unit vector.

    Args:
        positions (numpy.ndarray):
            (n, 3) connection-point positions. No point may lie at the centroid.

    Returns:
        An (n, 3) array of unit vectors in the original point order.
    '''
    centered = positions - positions.mean(axis=0)
    return centered / np.linalg.norm(centered, axis=1, keepdims=True)

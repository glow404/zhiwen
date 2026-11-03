"""
DIST2 Calculates squared distance between two sets of points.

Description
D = DIST2(X, C) takes two matrices of vectors and calculates the
squared Euclidean distance between them.  Both matrices must be of
the same column dimension.  If X has M rows and N columns, and C has
L rows and N columns, then the result has M rows and L columns.  The
I, Jth entry is the  squared distance from the Ith row of X to the
Jth row of C.

See also
GMMACTIV, KMEANS, RBFFWD

Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)
"""

import numpy as np
from typing import Union


def dist2(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Calculate squared Euclidean distance between two sets of points
    
    Args:
        x: First set of points (M x N matrix)
        c: Second set of points (L x N matrix)
        
    Returns:
        n2: Squared distance matrix (M x L)
        
    Raises:
        ValueError: If data dimensions don't match
    """
    ndata, dimx = x.shape
    ncentres, dimc = c.shape
    
    if dimx != dimc:
        raise ValueError('Data dimension does not match dimension of centres')
    
    # Calculate squared distances using broadcasting
    # (ones(ncentres, 1) * sum((x.^2)', 1))' + ones(ndata, 1) * sum((c.^2)',1) - 2.*(x*(c'))
    x_squared = np.sum(x**2, axis=1, keepdims=True)  # (ndata, 1)
    c_squared = np.sum(c**2, axis=1)  # (ncentres,)
    
    n2 = x_squared + c_squared - 2 * np.dot(x, c.T)
    
    return n2


if __name__ == "__main__":
    # Example usage
    x = np.random.rand(5, 3)  # 5 points in 3D
    c = np.random.rand(4, 3)  # 4 centers in 3D
    
    distances = dist2(x, c)
    print(f"Input x shape: {x.shape}")
    print(f"Input c shape: {c.shape}")
    print(f"Output distances shape: {distances.shape}")
    print(f"Distance matrix:\n{distances}")
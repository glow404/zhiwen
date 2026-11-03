"""
COORDINATION TRANSFORM FUNCTION

Usage:  [ Tnew ] = transform2( T, alpha );

Argument:   T     - Transformed Minutiae
            alpha - Rotation angle
              
Returns:    Tnew  - T with new coordinations

Vahid. K. Alilou
Department of Computer Engineering
The University of Semnan

July 2013
"""

import numpy as np
from typing import Union


def transform2(T: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply additional rotation transformation to minutiae
    
    Args:
        T: Transformed minutiae (N x 4 array)
        alpha: Rotation angle in radians
        
    Returns:
        Tnew: T with new coordinates after rotation
    """
    Count = T.shape[0]
    Tnew = np.zeros((Count, 4))
    
    # Transformation matrix
    R = np.array([
        [np.cos(alpha), np.sin(alpha), 0, 0],
        [-np.sin(alpha), np.cos(alpha), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    for i in range(Count):
        # Subtract rotation from angle component (angle is in column 3)
        B = T[i, :] - np.array([0, 0, 0, alpha])
        # Apply transformation matrix
        Tnew[i, :] = R @ B
    
    return Tnew


if __name__ == "__main__":
    # Example usage
    # Create sample transformed minutiae data
    T = np.array([
        [10, 20, 0.5, 1],
        [30, 40, 1.2, 2],
        [50, 60, 2.1, 1]
    ])
    
    # Apply additional rotation of 0.3 radians
    alpha = 0.3
    Tnew = transform2(T, alpha)
    
    print(f"Original transformed minutiae:\n{T}")
    print(f"After additional rotation of {alpha} radians:\n{Tnew}")
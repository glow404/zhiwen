"""
COORDINATION TRANSFORM FUNCTION

Usage:  [ T ] = transform( M, i );

Argument:   M -  Extracted Minutiae 
            i -  Index of reference minutia
              
Returns:    T -  M with new coordinations

Vahid. K. Alilou
Department of Computer Engineering
The University of Semnan

July 2013
"""

import numpy as np
from typing import Union


def transform(M: np.ndarray, i: int) -> np.ndarray:
    """
    Transform minutiae coordinates relative to a reference minutia
    
    Args:
        M: Extracted minutiae (N x 4 array: x, y, type, theta)
        i: Index of reference minutia (0-based)
        
    Returns:
        T: Transformed minutiae with new coordinates
    """
    Count = M.shape[0]
    
    # Get reference point coordinates and orientation
    XRef = M[i, 0]
    YRef = M[i, 1] 
    ThRef = M[i, 3]  # Assuming theta is in column 3 (0-based indexing)
    
    T = np.zeros((Count, 4))
    
    # Transformation matrix
    R = np.array([
        [np.cos(ThRef), np.sin(ThRef), 0],
        [-np.sin(ThRef), np.cos(ThRef), 0],
        [0, 0, 1]
    ])
    
    for idx in range(Count):
        # Create vector for transformation
        B = np.array([
            M[idx, 0] - XRef,
            M[idx, 1] - YRef, 
            M[idx, 3] - ThRef
        ])
        
        # Apply transformation
        transformed = R @ B
        T[idx, 0] = transformed[0]  # x
        T[idx, 1] = transformed[1]  # y
        T[idx, 2] = M[idx, 2]       # type (unchanged)
        T[idx, 3] = transformed[2]  # theta
    
    return T


if __name__ == "__main__":
    # Example usage
    # Create sample minutiae data (x, y, type, theta)
    M = np.array([
        [10, 20, 1, 0.5],
        [30, 40, 2, 1.2],
        [50, 60, 1, 2.1],
        [70, 80, 2, 0.8]
    ])
    
    # Transform relative to minutia at index 1
    T = transform(M, 1)
    print(f"Original minutiae shape: {M.shape}")
    print(f"Transformed minutiae shape: {T.shape}")
    print(f"Original minutiae:\n{M}")
    print(f"Transformed minutiae:\n{T}")
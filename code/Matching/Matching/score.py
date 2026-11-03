"""
TRANSFORMED MINUTIAE MATCHING SCORE

Usage:  [ si ] = score( T1, T2 );

Argument:   T1 -  First  Transformed Minutiae 
            T2 -  Second Transformed Minutiae
              
Returns:    sm - Similarity Measure

Vahid. K. Alilou
Department of Computer Engineering
The University of Semnan

July 2013
"""

import numpy as np
from typing import Union


def score(T1: np.ndarray, T2: np.ndarray) -> float:
    """
    Calculate similarity score between two sets of transformed minutiae
    
    Args:
        T1: First transformed minutiae (N x 4 array)
        T2: Second transformed minutiae (M x 4 array)
        
    Returns:
        sm: Similarity measure (float between 0 and 1)
    """
    Count1, Count2 = T1.shape[0], T2.shape[0]
    n = 0
    T = 15   # Threshold for distance
    TT = 14  # Threshold for theta (degrees)
    
    for i in range(Count1):
        Found = False
        j = 0
        
        while not Found and j < Count2:
            # Calculate Euclidean distance
            dx = T1[i, 0] - T2[j, 0]
            dy = T1[i, 1] - T2[j, 1]
            d = np.sqrt(dx**2 + dy**2)
            
            if d < T:
                # Calculate angular difference (angle is in column 3, not 2)
                DTheta = abs(T1[i, 3] - T2[j, 3]) * 180 / np.pi
                DTheta = min(DTheta, 360 - DTheta)
                
                if DTheta < TT:
                    n += 1  # Increase score
                    Found = True
            
            j += 1
    
    # Calculate similarity index
    sm = np.sqrt(n**2 / (Count1 * Count2))
    return sm


if __name__ == "__main__":
    # Example usage
    # Create sample minutiae data (x, y, theta, type)
    T1 = np.array([
        [10, 20, 0.5, 1],
        [30, 40, 1.2, 2],
        [50, 60, 2.1, 1]
    ])
    
    T2 = np.array([
        [12, 22, 0.6, 1],
        [32, 42, 1.1, 2],
        [70, 80, 2.8, 1]
    ])
    
    similarity = score(T1, T2)
    print(f"Similarity score: {similarity:.4f}")
    print(f"T1 shape: {T1.shape}, T2 shape: {T2.shape}")
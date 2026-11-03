"""
MINUTIAE MATCHING FUNCTION

Usage:  [ si ] = match( M1, M2 );

Argument:   M1 -  First  Minutiae 
            M2 -  Second Minutiae
              
Returns:    si - Similarity Index

Vahid. K. Alilou
Department of Computer Engineering
The University of Semnan

July 2013
"""

import numpy as np
from typing import Union
from transform import transform
from transform2 import transform2
from score import score


def match(M1: np.ndarray, M2: np.ndarray) -> float:
    """
    Calculate similarity index between two minutiae sets
    
    Args:
        M1: First minutiae set (N x 4 array)
        M2: Second minutiae set (M x 4 array)
        
    Returns:
        si: Similarity index (float between 0 and 1)
    """
    Count1, Count2 = M1.shape[0], M2.shape[0]
    
    # Initialize maximum similarity
    si = 0
    
    # Limit the number of reference points to improve performance
    max_ref_points = min(20, Count1, Count2)  # Limit to 20 reference points
    
    # Use larger rotation steps to reduce computation
    rotation_steps = np.arange(0, 2*np.pi, np.pi/3)  # 60-degree steps (6 rotations)
    
    total_iterations = max_ref_points * max_ref_points * len(rotation_steps)
    current_iteration = 0
    
    # Try limited reference points from M1
    for i in range(min(max_ref_points, Count1)):
        # Transform M1 relative to minutia i
        T1 = transform(M1, i)
        
        # Try limited reference points from M2
        for j in range(min(max_ref_points, Count2)):
            # Transform M2 relative to minutia j
            T2 = transform(M2, j)
            
            # Try different rotation angles
            for alpha in rotation_steps:
                current_iteration += 1
                
                # Apply additional rotation to T2
                T2_rotated = transform2(T2, alpha)
                
                # Calculate similarity score
                s = score(T1, T2_rotated)
                
                # Keep track of maximum similarity
                if s > si:
                    si = s
                
                # Early termination if very good match found
                if si > 0.8:
                    return si
                
                # Update progress every 100 iterations
                if current_iteration % 100 == 0:
                    progress = (current_iteration / total_iterations) * 100
                    print(f"\rMatching progress: {progress:.1f}%", end='', flush=True)
    
    print()  # New line after progress
    return si


if __name__ == "__main__":
    # Example usage
    # Create sample minutiae data (x, y, type, theta)
    M1 = np.array([
        [10, 20, 1, 0.5],
        [30, 40, 2, 1.2],
        [50, 60, 1, 2.1],
        [70, 80, 2, 0.8]
    ])
    
    M2 = np.array([
        [12, 22, 1, 0.6],
        [32, 42, 2, 1.1],
        [52, 62, 1, 2.0],
        [72, 82, 2, 0.9]
    ])
    
    similarity_index = match(M1, M2)
    print(f"Similarity index between M1 and M2: {similarity_index:.4f}")
    print(f"M1 shape: {M1.shape}, M2 shape: {M2.shape}")
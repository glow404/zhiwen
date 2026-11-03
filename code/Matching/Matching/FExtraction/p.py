"""
Sub function for getting pixel values based on 8-connectivity chart:
 4 | 3 | 2
 5 |   | 1
 6 | 7 | 8
"""

import numpy as np
from typing import Tuple


def p(img: np.ndarray, x: int, y: int, i: int) -> Tuple[int, int, int]:
    """
    Get pixel value and coordinates based on 8-connectivity chart
    
    Args:
        img: Input image array
        x: X coordinate (column)
        y: Y coordinate (row)
        i: Direction index (1-8, or 9 which is same as 1)
        
    Returns:
        Tuple of (pixel_value, X_coordinate, Y_coordinate)
    """
    if i == 1 or i == 9:
        Y = y
        X = x + 1
        j = img[y, x + 1] if X < img.shape[1] else 0
    elif i == 2:
        Y = y - 1
        X = x + 1
        j = img[y - 1, x + 1] if Y >= 0 and X < img.shape[1] else 0
    elif i == 3:
        Y = y - 1
        X = x
        j = img[y - 1, x] if Y >= 0 else 0
    elif i == 4:
        Y = y - 1
        X = x - 1
        j = img[y - 1, x - 1] if Y >= 0 and X >= 0 else 0
    elif i == 5:
        Y = y
        X = x - 1
        j = img[y, x - 1] if X >= 0 else 0
    elif i == 6:
        Y = y + 1
        X = x - 1
        j = img[y + 1, x - 1] if Y < img.shape[0] and X >= 0 else 0
    elif i == 7:
        Y = y + 1
        X = x
        j = img[y + 1, x] if Y < img.shape[0] else 0
    elif i == 8:
        Y = y + 1
        X = x + 1
        j = img[y + 1, x + 1] if Y < img.shape[0] and X < img.shape[1] else 0
    else:
        raise ValueError(f"Invalid direction index: {i}. Must be 1-8 or 9.")
    
    return int(j), X, Y


if __name__ == "__main__":
    # Example usage
    test_img = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])
    
    # Test from center position (1, 1)
    for direction in range(1, 9):
        val, x_coord, y_coord = p(test_img, 1, 1, direction)
        print(f"Direction {direction}: value={val}, coords=({x_coord}, {y_coord})")
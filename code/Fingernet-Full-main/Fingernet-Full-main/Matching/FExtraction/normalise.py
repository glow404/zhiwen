"""
NORMALISE - Normalises image values to 0-1, or to desired mean and variance

Usage:
            n = normalise(im)

Offsets and rescales image so that the minimum value is 0
and the maximum value is 1.  Result is returned in n.  If the image is
colour the image is converted to HSV and the value/intensity component
is normalised to 0-1 before being converted back to RGB.


            n = normalise(im, reqmean, reqvar)

Arguments:  im      - A grey-level input image.
            reqmean - The required mean value of the image.
            reqvar  - The required variance of the image.

Offsets and rescales image so that it has mean reqmean and variance
reqvar.  Colour images cannot be normalised in this manner.

Copyright (c) 1996-2005 Peter Kovesi
School of Computer Science & Software Engineering
The University of Western Australia
http://www.csse.uwa.edu.au/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

The Software is provided "as is", without warranty of any kind.

January 2005 - modified to allow desired mean and variance
"""

import numpy as np
from typing import Optional, Union
import colorsys


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to HSV color space"""
    return np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in rgb.reshape(-1, 3)]).reshape(rgb.shape)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV to RGB color space"""
    return np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in hsv.reshape(-1, 3)]).reshape(hsv.shape)


def normalise(im: np.ndarray, reqmean: Optional[float] = None, 
              reqvar: Optional[float] = None) -> np.ndarray:
    """
    Normalise image values to 0-1, or to desired mean and variance
    
    Args:
        im: Input image (grayscale or color)
        reqmean: Required mean value (optional)
        reqvar: Required variance (optional)
        
    Returns:
        n: Normalised image
        
    Raises:
        ValueError: If invalid number of arguments or color image with mean/var normalisation
    """
    if reqmean is not None and reqvar is None:
        raise ValueError('Both reqmean and reqvar must be specified together')
    if reqmean is None and reqvar is not None:
        raise ValueError('Both reqmean and reqvar must be specified together')
    
    # Convert to float if needed
    if im.dtype != np.float64:
        im = im.astype(np.float64)
    
    if reqmean is None:  # Normalise 0 - 1
        if len(im.shape) == 3:  # Assume colour image
            # Convert RGB to HSV
            hsv = rgb_to_hsv(im)
            v = hsv[:, :, 2]  # Value component
            
            # Normalise value component
            v = v - np.min(v)
            if np.max(v) > 0:
                v = v / np.max(v)
            
            hsv[:, :, 2] = v
            n = hsv_to_rgb(hsv)
        else:  # Assume greyscale
            n = im - np.min(im)
            if np.max(n) > 0:
                n = n / np.max(n)
            else:
                n = im.copy()
    
    else:  # Normalise to desired mean and variance
        if len(im.shape) == 3:  # colour image?
            raise ValueError('Cannot normalise colour image to desired mean and variance')
        
        # Zero mean, unit std dev
        im_centered = im - np.mean(im)
        std_dev = np.std(im)
        
        if std_dev > 0:
            im_normalized = im_centered / std_dev
        else:
            im_normalized = im_centered
        
        n = reqmean + im_normalized * np.sqrt(reqvar)
    
    return n


if __name__ == "__main__":
    # Example usage
    # Test with grayscale image
    test_gray = np.random.rand(100, 100) * 255
    normalized_gray = normalise(test_gray)
    print(f"Original grayscale range: [{test_gray.min():.2f}, {test_gray.max():.2f}]")
    print(f"Normalized grayscale range: [{normalized_gray.min():.2f}, {normalized_gray.max():.2f}]")
    
    # Test with mean and variance
    normalized_mv = normalise(test_gray, reqmean=100, reqvar=50)
    print(f"Mean/Var normalized - Mean: {np.mean(normalized_mv):.2f}, Var: {np.var(normalized_mv):.2f}")
    
    # Test with color image
    test_color = np.random.rand(50, 50, 3)
    normalized_color = normalise(test_color)
    print(f"Color image normalized shape: {normalized_color.shape}")
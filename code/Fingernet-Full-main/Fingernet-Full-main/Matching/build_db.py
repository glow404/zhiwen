"""
BUILDING FINGERPRINT MINUTIAE DATABASE

Usage:  build_db(ICount, JCount);

Argument:   ICount -  Number of FingerPrints 
            JCount -  Number of Images Per FingerPrint

Vahid. K. Alilou
Department of Computer Engineering
The University of Semnan

July 2013
"""

import numpy as np
import pickle
from PIL import Image
import os
from ext_finger import ext_finger

def build_db(ICount, JCount):
    """
    Build fingerprint minutiae database
    
    Parameters:
    ICount - Number of FingerPrints 
    JCount - Number of Images Per FingerPrint
    """
    p = 0
    ff = {}
    
    for i in range(1, ICount + 1):
        for j in range(1, JCount + 1):
            filename = f'FVC2002/DB1_B/10{i}_{j}.tif'
            
            try:
                # Read image
                img = Image.open(filename)
                img = np.array(img)
                p += 1
                
                # Convert to grayscale if color image
                if len(img.shape) == 3:
                    img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale
                
                print(f'extracting features from {filename} ...')
                
                # Extract minutiae
                minutiae, _, _, _, _, _, _ = ext_finger(img, 1)
                ff[p] = minutiae
                
            except FileNotFoundError:
                print(f'Warning: File {filename} not found, skipping...')
                continue
            except Exception as e:
                print(f'Error processing {filename}: {e}')
                continue
    
    # Save database
    with open('db.pkl', 'wb') as f:
        pickle.dump(ff, f)
    
    print(f'Database built successfully with {len(ff)} fingerprints')
    return ff


# Example usage
if __name__ == "__main__":
    # Test the function with small parameters
    try:
        # Build database with 2 fingerprints, 1 image each
        db = build_db(2, 1)
        print("Database building completed successfully")
        print(f"Number of entries in database: {len(db)}")
    except Exception as e:
        print(f"Error during database building: {e}")
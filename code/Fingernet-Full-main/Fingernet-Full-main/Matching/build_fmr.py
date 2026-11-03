"""
GENERATING FMR AND FNMR DATA

Usage:  build_fmr();

Argument:   Nothing

Vahid. K. Alilou
Department of Computer Engineering
The University of Semnan

July 2013
"""

import numpy as np
import pickle
from match import match

def build_fmr():
    """
    Generate FMR (False Match Rate) and FNMR (False Non-Match Rate) data
    """
    # Load database
    try:
        with open('db.pkl', 'rb') as f:
            ff = pickle.load(f)
    except FileNotFoundError:
        print("Error: Database file 'db.pkl' not found. Please run build_db first.")
        return None, None
    
    P = 72
    fmr = np.zeros((100, P))
    fnmr = np.zeros((100, P))
    
    for p in range(1, P + 1):
        if p not in ff:
            print(f"Warning: Entry {p} not found in database, skipping...")
            continue
            
        ffnew = ff[p]
        S = np.zeros(72)
        
        for i in range(1, 73):
            second = f'10{((i-1)//8)+1}_{((i-1)%8)+1}'
            if i in ff:
                S[i-1] = match(ffnew, ff[i])
            else:
                S[i-1] = 0
            print(f'\b\b{i}', end='', flush=True)
        
        # Ground truth indices
        gt = np.array([1, 2, 3, 4, 5, 6, 7, 8]) + (((p-1)//8) * 8)
        
        k = 0
        for a in np.arange(0.01, 1.01, 0.01):
            s = 0
            g = np.where(S > a)[0] + 1  # Convert to 1-based indexing
            
            for i in range(len(gt)):
                s += np.sum(g == gt[i])
            
            fmr[k, p-1] = (len(g) - s) / 72
            fnmr[k, p-1] = 1 - s / 8
            k += 1
    
    # Save results
    np.save('fmr.npy', fmr)
    np.save('fnmr.npy', fnmr)
    
    # Also save as pickle for compatibility
    with open('fmr.pkl', 'wb') as f:
        pickle.dump(fmr, f)
    with open('fnmr.pkl', 'wb') as f:
        pickle.dump(fnmr, f)
    
    print("\nFMR and FNMR data generated successfully")
    return fmr, fnmr


# Example usage
if __name__ == "__main__":
    try:
        fmr, fnmr = build_fmr()
        if fmr is not None and fnmr is not None:
            print(f"FMR shape: {fmr.shape}")
            print(f"FNMR shape: {fnmr.shape}")
            print("FMR and FNMR generation completed successfully")
    except Exception as e:
        print(f"Error during FMR/FNMR generation: {e}")
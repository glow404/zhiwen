"""
Main script for calculating and plotting FMR and FNMR

This script builds FMR and FNMR data and plots the results.
The FMR/FNMR building process can take about 2 hours.

Vahid. K. Alilou
Department of Computer Engineering
The University of Semnan

July 2013
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from build_fmr import build_fmr

def main_calc_fmr():
    """
    Main function to calculate and plot FMR and FNMR
    """
    print("Starting FMR and FNMR calculation...")
    
    # BUILDING FMR AND FNMR, THIS WILL TAKE ABOUT 2 HOURS
    # Uncomment the line below to build FMR/FNMR data
    # build_fmr()
    
    # PLOT FMR & FNMR
    try:
        # Try to load numpy files first
        if os.path.exists('fmr.npy') and os.path.exists('fnmr.npy'):
            fmr = np.load('fmr.npy')
            fnmr = np.load('fnmr.npy')
        # Fallback to pickle files
        elif os.path.exists('fmr.pkl') and os.path.exists('fnmr.pkl'):
            with open('fmr.pkl', 'rb') as f:
                fmr = pickle.load(f)
            with open('fnmr.pkl', 'rb') as f:
                fnmr = pickle.load(f)
        else:
            print("FMR and FNMR data files not found. Building them now...")
            fmr, fnmr = build_fmr()
            if fmr is None or fnmr is None:
                print("Failed to build FMR/FNMR data")
                return
    except Exception as e:
        print(f"Error loading FMR/FNMR data: {e}")
        return
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.hold = True  # For compatibility
    
    # Calculate average FMR and FNMR
    afmr = np.mean(fmr, axis=1)
    afnmr = np.mean(fnmr, axis=1)
    
    # Create x-axis values
    x_values = np.arange(0.01, 1.01, 0.01)
    
    # Plot FNMR
    plt.plot(x_values, afnmr, 'b-', linewidth=2, label='FNMR')
    plt.text(0.60, 0.3, '← FNMR', horizontalalignment='left', fontsize=12)
    
    # Plot FMR
    plt.plot(x_values, afmr, 'r-', linewidth=2, label='FMR')
    plt.text(0.27, 0.5, 'FMR →', horizontalalignment='right', fontsize=12)
    
    # Plot Zero FMR line
    plt.plot([0.48, 0.48], [0.01, 0.88], '-.m', linewidth=1)
    plt.text(0.42, 0.92, 'Zero FMR', fontsize=12)
    
    # Mark EER point
    plt.text(0.37, 0.09, 'EER', fontsize=12)
    
    # Set labels and formatting
    plt.xlabel('Similarity Score', fontsize=12)
    plt.ylabel('Rate', fontsize=12)
    plt.title('FMR and FNMR Curves', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set axis limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
    # Calculate and display EER
    # Find Equal Error Rate (where FMR ≈ FNMR)
    diff = np.abs(afmr - afnmr)
    eer_idx = np.argmin(diff)
    eer_value = (afmr[eer_idx] + afnmr[eer_idx]) / 2
    eer_threshold = x_values[eer_idx]
    
    print(f"Equal Error Rate (EER): {eer_value:.4f}")
    print(f"EER Threshold: {eer_threshold:.2f}")
    
    return fmr, fnmr, afmr, afnmr


# Example usage
if __name__ == "__main__":
    try:
        fmr, fnmr, afmr, afnmr = main_calc_fmr()
        print("FMR and FNMR calculation and plotting completed successfully")
    except Exception as e:
        print(f"Error during FMR/FNMR calculation: {e}")
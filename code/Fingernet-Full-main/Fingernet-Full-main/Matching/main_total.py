"""
Main script for complete fingerprint matching system

This script builds a fingerprint template database and performs
comprehensive matching against all templates.

Vahid. K. Alilou
Department of Computer Engineering
The University of Semnan

July 2013
"""

import numpy as np
from PIL import Image
import pickle
import os
import sys
from build_db import build_db
from ext_finger import ext_finger
from match import match

def main_total(filename='temp/101_1.bmp', threshold=0.48, build_new_db=False):
    """
    Complete fingerprint matching system
    
    Parameters:
    filename - Input fingerprint image file
    threshold - Matching threshold for determining matches (default: 0.48)
    build_new_db - Whether to build a new database (default: False)
    
    Returns:
    matched_fingerprints - Indices of matched fingerprints
    scores - All similarity scores
    """
    
    # BUILD FINGERPRINT TEMPLATE DATABASE
    if build_new_db:
        print("Building fingerprint template database...")
        print("THIS WILL TAKE ABOUT 30 MINUTES")
        build_db(9, 8)
    
    # Load existing database
    try:
        if os.path.exists('db.pkl'):
            with open('db.pkl', 'rb') as f:
                ff = pickle.load(f)
        else:
            print("Database not found. Building new database...")
            build_db(9, 8)
            with open('db.pkl', 'rb') as f:
                ff = pickle.load(f)
    except Exception as e:
        print(f"Error loading database: {e}")
        return None, None
    
    # EXTRACT FEATURES FROM AN ARBITRARY FINGERPRINT
    try:
        img = Image.open(filename)
        img = np.array(img)
        
        # Convert to grayscale if color image
        if len(img.shape) == 3:
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale
            
        print(f'Extracting features from {filename} ...')
        
        # Extract minutiae from the input image
        minutiae, _, _, _, _, _, _ = ext_finger(img, 1)
        ffnew = minutiae
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None
    except Exception as e:
        print(f"Error processing image '{filename}': {e}")
        return None, None
    
    # FOR EACH FINGERPRINT TEMPLATE, CALCULATE MATCHING SCORE IN COMPARISON WITH FIRST ONE
    print("\nStarting comprehensive matching...")
    S = np.zeros(72)
    
    for i in range(1, 73):
        if i in ff:
            second = f'10{((i-1)//8)+1}_{((i-1)%8)+1}'
            print(f'Computing similarity between {filename} and {second} from FVC2002 : ', end='')
            
            try:
                score = match(ffnew, ff[i])
                S[i-1] = score
                print(f'{score:.4f}')
                
            except Exception as e:
                print(f'Error - {e}')
                S[i-1] = 0
        else:
            print(f'Template {i} not found in database')
            S[i-1] = 0
    
    # OFFER MATCHED FINGERPRINTS
    matched_indices = np.where(S > threshold)[0] + 1  # Convert to 1-based indexing
    matched_fingerprints = matched_indices.tolist()
    
    print(f"\n=== MATCHING RESULTS ===")
    print(f"Threshold: {threshold}")
    print(f"Matched Fingerprints: {matched_fingerprints}")
    print(f"Number of matches: {len(matched_fingerprints)}")
    
    if len(matched_fingerprints) > 0:
        print(f"Best match score: {np.max(S):.4f}")
        best_match_idx = np.argmax(S) + 1
        best_match_name = f'10{((best_match_idx-1)//8)+1}_{((best_match_idx-1)%8)+1}'
        print(f"Best match: {best_match_name} (Index: {best_match_idx})")
    else:
        print("No matches found above threshold")
    
    # Display score statistics
    print(f"\n=== SCORE STATISTICS ===")
    print(f"Average score: {np.mean(S):.4f}")
    print(f"Standard deviation: {np.std(S):.4f}")
    print(f"Min score: {np.min(S):.4f}")
    print(f"Max score: {np.max(S):.4f}")
    
    return matched_fingerprints, S


def analyze_scores(scores, filename):
    """
    Analyze and visualize matching scores
    
    Parameters:
    scores - Array of similarity scores
    filename - Name of the query fingerprint
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot scores
        plt.subplot(2, 1, 1)
        plt.plot(range(1, len(scores) + 1), scores, 'b-o', markersize=3)
        plt.axhline(y=0.48, color='r', linestyle='--', label='Threshold (0.48)')
        plt.xlabel('Template Index')
        plt.ylabel('Similarity Score')
        plt.title(f'Matching Scores for {filename}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot histogram
        plt.subplot(2, 1, 2)
        plt.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.48, color='r', linestyle='--', label='Threshold (0.48)')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


# Example usage
if __name__ == "__main__":
    try:
        print("=== Complete Fingerprint Matching System ===")
        
        # Run complete matching
        matched_fps, scores = main_total('temp/101_1.bmp', 0.48, False)
        
        if matched_fps is not None and scores is not None:
            print("\nMatching completed successfully")
            
            # Analyze results if matplotlib is available
            try:
                analyze_scores(scores, '101_1.bmp')
            except:
                pass
            
            # Save results
            results = {
                'matched_fingerprints': matched_fps,
                'scores': scores,
                'threshold': 0.48,
                'query_file': '101_1.bmp'
            }
            
            with open('matching_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            
            print("Results saved to 'matching_results.pkl'")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
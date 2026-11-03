"""
输入单个指纹然后与数据库中的模板进行匹配，返回相似度的分数
"""

import numpy as np
from PIL import Image
import pickle
import os
from ext_finger import ext_finger
from match import match
from tqdm import tqdm

def main_single(filename='FVC2002/DB1_B/101_1.tif', template_index=2, show_details=True):
    """
    Extract features from a fingerprint and match with database template
    
    Parameters:
    filename - Input fingerprint image file
    template_index - Index of template fingerprint in database (default: 2)
    show_details - Whether to show detailed matching results (default: True)
    
    Returns:
    similarity_score - Matching score between the two fingerprints
    """
    
    # EXTRACT FEATURES FROM AN ARBITRARY FINGERPRINT
    try:
        img = Image.open(filename)
        img = np.array(img)
        
        # Convert to grayscale if color image
        if len(img.shape) == 3:
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale
            
        print(f'Extracting features from {filename} ...')
        
        # Extract minutiae from the input image
        minutiae, _, _, _, _, _, _ = ext_finger(img, 1 if show_details else 0)
        ffnew = minutiae
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error processing image '{filename}': {e}")
        return None
    
    # GET FEATURES OF AN ARBITRARY FINGERPRINT FROM THE TEMPLATE AND MATCH IT WITH FIRST ONE
    try:
        # Load database
        if os.path.exists('db.pkl'):
            with open('db.pkl', 'rb') as f:
                ff = pickle.load(f)
        else:
            print("Error: Database file 'db.pkl' not found. Please run build_db first.")
            return None
        
        # Check if template index exists in database
        if template_index not in ff:
            print(f"Error: Template index {template_index} not found in database.")
            return None
        
        # Generate template filename for display
        i = template_index
        second = f'10{((i-1)//8)+1}_{((i-1)%8)+1}'
        
        print(f'Computing similarity between {filename} and {second} from FVC2002')
        
        # Perform matching with progress indication
        print("Performing minutiae matching...")
        with tqdm(total=1, desc="Matching", unit="comparison") as pbar:
            S = match(ffnew, ff[i])
            pbar.update(1)
        
        print(f'Similarity score: {S:.4f}')
        
        return S
        
    except Exception as e:
        print(f"Error during matching: {e}")
        return None


def batch_match(filename, show_details=False):
    """
    Match input fingerprint against all templates in database
    
    Parameters:
    filename - Input fingerprint image file
    show_details - Whether to show detailed matching results
    
    Returns:
    scores - Array of similarity scores for all templates
    """
    
    # Extract features from input image
    try:
        img = Image.open(filename)
        img = np.array(img)
        
        if len(img.shape) == 3:
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
            
        print(f'Extracting features from {filename} ...')
        minutiae, _, _, _, _, _, _ = ext_finger(img, 0)  # No display for batch
        ffnew = minutiae
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    
    # Load database
    try:
        if os.path.exists('db.pkl'):
            with open('db.pkl', 'rb') as f:
                ff = pickle.load(f)
        else:
            print("Error: Database not found.")
            return None
        
        scores = []
        # 添加进度条显示匹配进度
        template_keys = list(ff.keys())
        print(f"Matching against {len(template_keys)} templates in database...")
        
        for i in tqdm(template_keys, desc="Matching progress", unit="template"):
            score = match(ffnew, ff[i])  # No display for batch
            scores.append(score)
            if show_details:
                second = f'10{((i-1)//8)+1}_{((i-1)%8)+1}'
                tqdm.write(f'Score with {second}: {score:.4f}')  # 使用tqdm.write避免进度条冲突
        
        return np.array(scores)
        
    except Exception as e:
        print(f"Error during batch matching: {e}")
        return None


# Example usage
if __name__ == "__main__":
    try:
        # Single matching example
        print("=== Single Fingerprint Matching ===")
        score = main_single('FVC2002/DB1_B/101_1.tif', 1, True)
        
        if score is not None:
            print(f"\nMatching completed successfully")
            print(f"Final similarity score: {score:.4f}")
            
            # Interpret the score
            if score > 0.5:
                print("Result: MATCH (High similarity)")
            elif score > 0.3:
                print("Result: POSSIBLE MATCH (Medium similarity)")
            else:
                print("Result: NO MATCH (Low similarity)")
        
        # Batch matching example (uncomment to enable)
        print("\n=== Batch Matching ===")
        scores = batch_match('FVC2002/DB1_B/101_1.tif', True)
        if scores is not None:
            print(f"Best match score: {np.max(scores):.4f}")
            print(f"Average score: {np.mean(scores):.4f}")
            print(f"Total templates matched: {len(scores)}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
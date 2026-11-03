"""
PLOT DATA

Usage:  plot_data(X, y);

Argument:   X -  Data Points (Nx3 array: [x, y, orientation])
            y -  Plot Style (1 for blue, 2 for red, ...)

Vahid. K. Alilou
Department of Computer Engineering
The University of Semnan

July 2013
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, y):
    """
    Plot minutiae data points with orientation lines
    
    Parameters:
    X - Data points array (Nx3): [x_coord, y_coord, orientation]
    y - Plot style index (1 for blue, 2 for red, etc.)
    """
    
    if X is None or len(X) == 0:
        print("No data points to plot")
        return
    
    # Ensure X is a numpy array
    X = np.array(X)
    
    # Handle different input formats
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    if X.shape[1] < 3:
        print("Warning: Data points should have at least 3 columns [x, y, orientation]")
        return
    
    N = X.shape[0]
    r = 15  # Length of orientation line
    
    # Define color palettes
    pale = ['b+', 'r.', 'g.', 'y.', 'm.', 'c.']
    color = ['b', 'r', 'g', 'y', 'm', 'c']
    
    # Ensure y is within valid range
    y = max(1, min(y, len(pale)))
    style_idx = y - 1  # Convert to 0-based index
    
    # Set up the plot
    plt.figure(figsize=(10, 8))
    plt.hold(True) if hasattr(plt, 'hold') else None
    plt.axis('equal')
    
    # Plot each data point
    for i in range(N):
        x_coord = X[i, 0]
        y_coord = X[i, 1]
        orientation = X[i, 2]
        
        # Plot the minutiae point
        plt.plot(x_coord, y_coord, pale[style_idx], markersize=8)
        
        # Calculate orientation line endpoints
        MyX = [x_coord, x_coord + r * np.cos(orientation)]
        MyY = [y_coord, y_coord + r * np.sin(orientation)]
        
        # Plot the orientation line
        plt.plot(MyX, MyY, color[style_idx], linewidth=2)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Minutiae Plot (Style {y})')
    
    return plt.gca()


def plot_multiple_data(data_list, styles=None, labels=None, title="Multiple Minutiae Plot"):
    """
    Plot multiple sets of minutiae data
    
    Parameters:
    data_list - List of data arrays
    styles - List of style indices (default: [1, 2, 3, ...])
    labels - List of labels for legend
    title - Plot title
    """
    
    if not data_list:
        print("No data to plot")
        return
    
    plt.figure(figsize=(12, 10))
    plt.axis('equal')
    
    if styles is None:
        styles = list(range(1, len(data_list) + 1))
    
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(data_list))]
    
    # Plot each dataset
    for i, (data, style) in enumerate(zip(data_list, styles)):
        if data is not None and len(data) > 0:
            plot_data_overlay(data, style, labels[i])
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.legend()
    
    return plt.gca()


def plot_data_overlay(X, y, label=None):
    """
    Plot data as overlay (for multiple datasets)
    
    Parameters:
    X - Data points array
    y - Plot style index
    label - Label for legend
    """
    
    if X is None or len(X) == 0:
        return
    
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    if X.shape[1] < 3:
        return
    
    N = X.shape[0]
    r = 15
    
    pale = ['b+', 'r.', 'g.', 'y.', 'm.', 'c.']
    color = ['b', 'r', 'g', 'y', 'm', 'c']
    
    y = max(1, min(y, len(pale)))
    style_idx = y - 1
    
    # Plot points and orientation lines
    x_coords = X[:, 0]
    y_coords = X[:, 1]
    orientations = X[:, 2]
    
    # Plot all points at once
    plt.plot(x_coords, y_coords, pale[style_idx], markersize=8, label=label)
    
    # Plot orientation lines
    for i in range(N):
        MyX = [x_coords[i], x_coords[i] + r * np.cos(orientations[i])]
        MyY = [y_coords[i], y_coords[i] + r * np.sin(orientations[i])]
        plt.plot(MyX, MyY, color[style_idx], linewidth=1, alpha=0.7)


# Example usage
if __name__ == "__main__":
    # Test the function with sample data
    try:
        # Create sample minutiae data (x, y, orientation)
        sample_data1 = np.array([
            [100, 150, 0.5],
            [200, 200, 1.2],
            [150, 100, 2.1],
            [250, 180, 0.8],
            [180, 250, 1.5]
        ])
        
        sample_data2 = np.array([
            [120, 170, 0.3],
            [220, 220, 1.0],
            [170, 120, 1.9],
            [270, 200, 0.6]
        ])
        
        # Test single dataset plotting
        print("Plotting single dataset...")
        plt.subplot(1, 2, 1)
        plot_data(sample_data1, 1)
        plt.title("Single Dataset (Blue)")
        
        # Test multiple dataset plotting
        print("Plotting multiple datasets...")
        plt.subplot(1, 2, 2)
        plot_multiple_data([sample_data1, sample_data2], [1, 2], 
                          ["Dataset 1", "Dataset 2"], "Multiple Datasets")
        
        plt.tight_layout()
        plt.show()
        
        print("Plot data function test completed successfully")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
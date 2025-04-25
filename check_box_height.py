import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_boxes(label_dir, class_id):
    """
    Analyzes the distribution of bounding box dimensions for a specific class in YOLO format dataset.
    
    This function processes label files to calculate and visualize width/height statistics of bounding boxes
    for a specified class, helping identify potential annotation issues or dataset biases.

    Parameters:
    -----------
    label_dir : str
        Path to directory containing YOLO format label files (.txt)
        Each file should contain lines in format: <class_id> <x_center> <y_center> <width> <height>
        Coordinates should be normalized (0-1 range relative to image dimensions)
        
    class_id : int
        The class ID to analyze (must match the IDs used in your label files)

    Returns:
    --------
    dict
        Dictionary containing computed statistics:
        {
            'mean_width': float,  # Average width across all boxes (normalized 0-1)
            'mean_height': float, # Average height across all boxes (normalized 0-1)
            'max_width': float,   # Maximum observed width (normalized 0-1)
            'max_height': float,  # Maximum observed height (normalized 0-1)
            'std_width': float,   # Standard deviation of widths
            'std_height': float   # Standard deviation of heights
        }

    Outputs:
    --------
    Displays a matplotlib figure with two subplots:
    - Left: Histogram of normalized widths (0-1)
    - Right: Histogram of normalized heights (0-1)
    """
    widths = []
    heights = []
    
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
            
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                cls, xc, yc, w, h = map(float, parts)
                if int(cls) == class_id:
                    widths.append(w)
                    heights.append(h)
    
    # Calculate statistics
    stats = {
        'mean_width': np.mean(widths),
        'mean_height': np.mean(heights),
        'max_width': np.max(widths),
        'max_height': np.max(heights),
        'std_width': np.std(widths),
        'std_height': np.std(heights)
    }
    
    # Plot distributions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=50, alpha=0.7, color='blue')
    plt.title('Width Distribution')
    
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=50, alpha=0.7, color='red')
    plt.title('Height Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return stats

# Usage: Analyze class 2 boxes in training set
label_dir = r"C:\Users\user\Desktop\Machine Learning\INT4097\Project\New_img\train\labels"
stats = analyze_boxes(label_dir, class_id=2)
print(f"Box size statistics:\n{stats}")
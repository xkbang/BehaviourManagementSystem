'''
Check the instances of different classes if there are imbalance classes
'''

import os
from collections import defaultdict
import pandas as pd

def analyze_yolo_dataset(data_root):
    """
    Analyzes a YOLO dataset with structure:
    data/
      images/
      labels/
    """
    # Initialize counters
    class_instance_counts = defaultdict(int)
    images_per_class = defaultdict(int)
    total_images = 0

    # Iterate through all label files
    labels_dir = os.path.join(data_root, 'labels')
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            total_images += 1
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                classes_in_image = set()
                for line in f:
                    class_id = int(line.strip().split()[0])
                    class_instance_counts[class_id] += 1
                    classes_in_image.add(class_id)
                
                for class_id in classes_in_image:
                    images_per_class[class_id] += 1

    # Convert to pandas DataFrame for nice display
    stats = pd.DataFrame({
        'Class ID': list(class_instance_counts.keys()),
        'Total Instances': list(class_instance_counts.values()),
        'Images Containing Class': [images_per_class[c] for c in class_instance_counts.keys()]
    }).sort_values('Class ID')

    print(f"\nDataset Analysis for: {data_root}")
    print(f"Total Images: {total_images}")
    print(f"Classes Found: {len(class_instance_counts)}")
    print("\nDetailed Statistics:")
    print(stats.to_string(index=False))

    return stats

# Usage
path = r"C:\Users\user\Desktop\Machine Learning\INT4097\Project\New_img\data"
dataset_stats = analyze_yolo_dataset(path)  # Path to your data folder
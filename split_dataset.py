import os
import random
import shutil
from sklearn.model_selection import train_test_split

def split_yolo_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split YOLO dataset into train/val/test sets while maintaining directory structure
    
    Args:
        data_dir (str): Path to original dataset (contains images/, labels/)
        output_dir (str): Path to save split dataset
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set
        test_ratio (float): Proportion for test set
        seed (int): Random seed for reproducibility
    """
    # Validate ratios
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.001, "Ratios must sum to 1"
    
    # Create paths
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    
    # Get all image files with original extensions
    image_files = [f for f in os.listdir(images_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Split into train, val, test
    train_val, test = train_test_split(image_files, test_size=test_ratio, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio), 
                                random_state=seed)
    
    # Create output directory structure
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Copy files to respective directories
    def copy_files(filenames, split_name):
        for filename in filenames:
            # Get base name without extension
            base_name = os.path.splitext(filename)[0]
            
            # Source paths
            src_image = os.path.join(images_dir, filename)
            src_label = os.path.join(labels_dir, f"{base_name}.txt")
            
            # Destination paths
            dst_image = os.path.join(output_dir, split_name, 'images', filename)
            dst_label = os.path.join(output_dir, split_name, 'labels', f"{base_name}.txt")
            
            # Copy image
            if os.path.exists(src_image):
                shutil.copy2(src_image, dst_image)
            else:
                print(f"Warning: Missing image {src_image}")
            
            # Copy label
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                print(f"Warning: Missing label {src_label}")
    
    copy_files(train, 'train')
    copy_files(val, 'val')
    copy_files(test, 'test')
    
    # Copy classes file if exists
    classes_file = os.path.join(data_dir, 'classes.txt')
    if os.path.exists(classes_file):
        shutil.copy2(classes_file, output_dir)
    
    print(f"\nDataset split complete:")
    print(f"- Training samples: {len(train)}")
    print(f"- Validation samples: {len(val)}")
    print(f"- Test samples: {len(test)}")

# Example usage
if __name__ == "__main__":
    data_dir = r"C:\Users\user\Desktop\Machine Learning\INT4097\Project\New_img\data"
    output_dir = r"C:\Users\user\Desktop\Machine Learning\INT4097\Project\New_img\split_data"
    
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    
    split_yolo_dataset(data_dir, output_dir, train_ratio, val_ratio, test_ratio)
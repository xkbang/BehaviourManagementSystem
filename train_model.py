'''
Train the model

'''



from ultralytics import YOLO
import os

# 1. Verify your dataset structure
def verify_dataset(yaml_path):
    """
    Verifies the required dataset directory structure exists.
    
    Purpose:
    - Ensures all necessary training/validation directories exist before starting training
    - Prevents training failures due to missing data
    
    Parameters:
    -----------
    yaml_path : str
        Path to the data.yaml configuration file
        The function checks directories relative to this file's location
        
    Raises:
    -------
    FileNotFoundError
        If any required directory is missing
    """
    required_dirs = ['train/images', 'train/labels', 
                    'val/images', 'val/labels']
    for dir in required_dirs:
        if not os.path.exists(os.path.join(os.path.dirname(yaml_path), dir)):
            raise FileNotFoundError(f"Missing directory: {dir}")

# 2. Training configuration
def train_model():
    """
    Trains a YOLOv11 model with customized configuration.
    
    Workflow:
    1. Verifies dataset structure
    2. Loads pretrained weights
    3. Configures training parameters
    4. Runs training
    5. Exports best model
    
    Returns:
    --------
    train_results : object
        Training results object containing metrics and paths
    
    Outputs:
    --------
    - Trained model weights (best.pt, last.pt)
    - ONNX exported model
    - Training logs and metrics
    """
    # Path to your data.yaml (use raw string for Windows paths)
    data_yaml = r"C:\Users\user\Desktop\Machine Learning\INT4097\Project\Project_img\data.yaml"
    
    # Verify dataset first
    verify_dataset(data_yaml)
    
    # Load the medium-large model (YOLOv11n)
    model = YOLO('yolo11s.pt')  # Official pretrained weights
    
    # Training parameters
    train_results = model.train(
        data=data_yaml,
        epochs=150,  # Increased for better convergence
        batch=32,    # Adjust based on GPU memory (16GB+ recommended for batch=32)
        imgsz=640,
        device="0",  # GPU
        workers=8,
        optimizer="AdamW",
        lr0=0.002,   # Higher initial LR for YOLOv11
        weight_decay=0.0005,
        hsv_h=0.02,  # Stronger augmentation
        hsv_s=0.8,
        hsv_v=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.2,
        close_mosaic=10,  # Disable mosaic last 10 epochs
        name="yolo11l_custom",
        pretrained=True,
        patience=15,
        amp=True  # Mixed precision training
    )
    
    # 4. Post-Training Actions
    best_model = YOLO(train_results.save_dir / 'weights' / 'best.pt')
    
    # Export to ONNX/TensorRT
    best_model.export(format="onnx", dynamic=True, simplify=True)
    
    return train_results

if __name__ == "__main__":
    print("Starting YOLOv11 training...")
    results = train_model()
    
    print(f"""
    Training Complete!
    - Results saved to: {results.save_dir}
    - Best model: {results.save_dir}/weights/best.pt
    - Metrics: {results.save_dir}/results.csv
    """)
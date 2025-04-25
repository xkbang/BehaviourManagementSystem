from ultralytics import YOLO

def fine_tune_model():
    data_yaml = r"C:\Users\user\Desktop\Machine Learning\INT4097\Project\Project_img\data.yaml"
    model = YOLO('yolo11s.pt')
    
    
    # Define search space for hyperparameter tuning
    search_space = {
        "lr0": (1e-5, 1e-2),      # Initial learning rate 
        "weight_decay": (0.0001, 0.001),
        "hsv_h": (0.0, 0.1),      # Hue augmentation 
        "hsv_s": (0.5, 1.0),      # Saturation augmentation 
        "hsv_v": (0.3, 0.7),      # Value augmentation 
        "fliplr": (0.3, 0.7),     # Horizontal flip prob 
        "mosaic": (0.8, 1.0),     # Mosaic augmentation 
        "mixup": (0.1, 0.3),      # MixUp augmentation 
    }
    
    # Tune hyperparameters
    tune_results = model.tune(
        data=data_yaml,
        epochs=30,
        iterations=100,           # Number of optimization trials
        optimizer="AdamW",
        space=search_space,
        # Fixed parameters from original training config
        batch=32,
        imgsz=640,
        device="0",
        workers=8,
        flipud=0.5,
        copy_paste=0.2,
        close_mosaic=10,
        name="yolo11l_tuned",
        pretrained=True,
        patience=15,
        amp=True,
        plots=True,
        save=True,
        val=True,
    )
    
    return tune_results

if __name__ == "__main__":
    print("Starting YOLOv11 hyperparameter tuning...")
    results = fine_tune_model()
    print(f"""
    Tuning Complete!
    - Best model saved to: {results.save_dir}
    - Best hyperparameters: {results.best_params}
    - Exported ONNX model available in weights directory
    """)
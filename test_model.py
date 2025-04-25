'''
Test the model, You may input one image here


'''

from ultralytics import YOLO

# Load your trained model
model_path = r"C:\Users\user\Desktop\Machine Learning\INT4097\Project\runs1\content\runs\detect\yolo11l_custom\weights\best.pt"
model = YOLO(model_path)  # Path to your trained weights

img_path = r"C:\Users\user\Desktop\Machine Learning\INT4097\Project\Project_img\test\images\0e25cd69-image518.jpg"


# Run inference
results = model.predict(
    source=img_path,  # Can be image/video/folder/webcam
    conf=0.5,           # Confidence threshold
    iou=0.45,           # NMS IoU threshold
    imgsz=640,          # Inference size
    show=True,          # Display results
    save=True,          # Save results
    save_txt=True       # Save results as YOLO format labels
)
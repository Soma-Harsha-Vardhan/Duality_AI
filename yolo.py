# # -------------------------
# # YOLOv8-L with ViT backbone training on K: drive
# # -------------------------

# from ultralytics import YOLO
# import glob
# from PIL import Image
# import os
# import pandas as pd

# # -------------------------
# # 1️⃣ Redirect temp/cache to K: drive to avoid SSD
# # -------------------------
# os.environ['TMP'] = 'K:/temp'
# os.environ['TEMP'] = 'K:/temp'
# os.environ['TMPDIR'] = 'K:/temp'

# if not os.path.exists('D:/temp'):
#     os.makedirs('D:/temp')

# # -------------------------
# # 2️⃣ Load YOLOv8-Large (ViT backbone) model
# # -------------------------
# model = YOLO("yolov8l.pt")  # Large model, better mAP

# # -------------------------
# # 3️⃣ Train the model
# # -------------------------
# results = model.train(
#     data="data.yaml",                # Your dataset YAML
#     epochs=50,                       # More epochs for better accuracy
#     imgsz=256,                       # Larger images improve detection
#     batch=4,                         # Adjust according to your GPU memory
#     optimizer="AdamW",
#     lr0=1e-3,
#     patience=10,
#     device=0,                         
#     project="K:/YOLO_train_results", 
#     name="yolo_v8_safety_vit_l",      
#     save=True,                        
#     val=True,                         
#     plots=True,                        
#     save_period=5,                     
#     half=True,                        
#     augment=True                       # Use augmentation for better generalization
# )

# # -------------------------
# # 4️⃣ Function to display last few predictions
# # -------------------------
# def show_latest_predictions(run_name="yolo_v8_safety_vit_l", last_n=3):
#     pred_path = f"K:/YOLO_train_results/{run_name}/predict"
#     if not os.path.exists(pred_path):
#         print("No predictions found yet.")
#         return
#     images = sorted(glob.glob(f"{pred_path}/*.jpg"))[-last_n:]
#     for img_file in images:
#         img = Image.open(img_file)
#         img.show()

# show_latest_predictions()

# # -------------------------
# # 5️⃣ Evaluate the model and print metrics per class
# # -------------------------
# metrics = model.val()  # Run validation, returns detailed metrics object

# # Print summary of overall metrics
# metrics.box.print()  # Prints mAP, precision, recall

# # -------------------------
# # 6️⃣ Create clean per-class metrics table
# # -------------------------
# def class_metrics_table(metrics_obj):
#     """
#     Displays a clean Pandas table with mAP50, mAP50-95, Precision, Recall for each class.
#     """
#     # Extract class names
#     class_names = metrics_obj.names  # list of class names
#     results_dict = metrics_obj.box.to_dict()  # per-class metrics

#     # Build table
#     table = pd.DataFrame({
#         "Class": class_names,
#         "Precision": results_dict.get("P", [0]*len(class_names)),
#         "Recall": results_dict.get("R", [0]*len(class_names)),
#         "mAP@0.5": results_dict.get("mAP_50", [0]*len(class_names)),
#         "mAP@0.5:0.95": results_dict.get("mAP_50_95", [0]*len(class_names))
#     })

#     # Format as percentages
#     table[["Precision","Recall","mAP@0.5","mAP@0.5:0.95"]] = table[["Precision","Recall","mAP@0.5","mAP@0.5:0.95"]] * 100
#     return table

# # Display the table
# metrics_table = class_metrics_table(metrics)
# display(metrics_table)
from ultralytics import YOLO
import pandas as pd

def main():
    # Load YOLOv8x (extra-large) model
    model = YOLO("yolov8x.pt")

    # Train on your dataset
    results = model.train(
        data="data.yaml",   # path to dataset config
        epochs=20,         # change as needed
        imgsz=256,          # image size
        batch=16,           # reduce if GPU runs out of memory
        workers=4,
        name="yolov8x_custom"
    )

    # Validate model after training
    metrics = model.val()

    # Create a clean results table
    metrics_table = pd.DataFrame({
        "Metric": ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"],
        "Value": [
            round(metrics.box.map50, 3),   # mAP@0.5
            round(metrics.box.map, 3),     # mAP@0.5:0.95
            round(metrics.box.mp, 3),      # mean precision
            round(metrics.box.mr, 3)       # mean recall
        ]
    })

    print("\n=== Evaluation Results ===")
    print(metrics_table.to_string(index=False))


if __name__ == "__main__":
    main()

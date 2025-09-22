YOLOv8-ViT safety object detection project
The Project includes:
•	Project overview
•	Dataset details
•	Installation instructions
•	Training code example
•	Validation results with per-class metrics provided the images in the repository
•	Notes, tips, and contact info
# YOLOv8-ViT Safety Object Detection
This repository contains a professional **object detection project using YOLOv8 with a ViT (Vision Transformer) backbone**.  
The model detects safety-related objects such as fire extinguishers, oxygen tanks, helmets, first aid boxes, safety panels, fire alarms, and emergency phones.  
The project is optimized for limited GPU memory, supports per-class evaluation, and saves all outputs on external drives.
--
##🔹 Features
- **YOLOv8 with ViT backbone** for enhanced detection performance.  
- **Multi-class detection**: 7 safety-related object classes.  
- **Optimized for low VRAM GPUs**: mixed precision, small batch sizes, temp/cache redirect.  
- **Per-class metrics**: precision, recall, mAP@0.5, mAP@0.5:0.95 displayed in tabular form.  
- **Flexible dataset support**: easily replace with your own images.  
- **Training logs, predictions, and checkpoints** saved to external drive.  
---
## 🔹 Dataset
- **Structure**:
dataset/
├─ images/
│ ├─ train/
│ └─ val/
├─ labels/
│ ├─ train/
│ └─ val/
└─ data.yaml

```yaml
train: dataset/images/train
val: dataset/images/val
nc: 7
names: ["OxygenTank", "NitrogenTank", "FirstAidBox", "FireAlarm", "SafetySwitchPanel", "EmergencyPhone", "FireExtinguisher"]
________________________________________
🔹 Installation
# Clone the repository
git clone https://github.com/yourusername/yolov8-safety-vit.git
cd yolov8-safety-vit

# Create virtual environment
python -m venv yolovenv
yolovenv\Scripts\activate   # Windows
# source yolovenv/bin/activate  # Linux/macOS

# Install dependencies
pip install --upgrade pip
pip install ultralytics pillow matplotlib
________________________________________
🔹 Training the Model
from ultralytics import YOLO
import os

# Redirect temp/cache to external drive
os.environ['TMP'] = 'K:/temp'
os.environ['TEMP'] = 'K:/temp'
os.environ['TMPDIR'] = 'K:/temp'

if not os.path.exists('K:/temp'):
    os.makedirs('K:/temp')

# Load YOLOv8 extra-large model with ViT backbone
model = YOLO("yolov8x.pt")  # automatically downloads weights if not present

# Train model
results = model.train(
    data="data.yaml",
    epochs=20,
    imgsz=256,
    batch=4,
    optimizer="AdamW",
    lr0=1e-3,
    patience=10,
    device=0,
    project="K:/YOLO_train_results",
    name="yolo_v8_safety_vit_xl",
    save=True,
    val=True,
    plots=True,
    save_period=5,
    half=True,
    augment=True
)
________________________________________
🔹 Validation & Per-class Metricss
From your last training run (YOLOv8x, 20 epochs, 256px images, batch=4):
 
Note: Precision = Correct detections / Total predicted
Recall = Correct detections / Total actual
mAP@0.5 = mean Average Precision at IoU 0.5
mAP@0.5:0.95 = mean Average Precision averaged over IoU thresholds 0.5 to 0.95
________________________________________
🔹 Display Latest Predictions
from PIL import Image
import glob

def show_latest_predictions(run_name="yolo_v8_safety_vit_xl", last_n=3):
    pred_path = f"K:/YOLO_train_results/{run_name}/predict"
    if not os.path.exists(pred_path):
        print("No predictions found yet.")
        return
    images = sorted(glob.glob(f"{pred_path}/*.jpg"))[-last_n:]
    for img_file in images:
        img = Image.open(img_file)
        img.show()

show_latest_predictions()
________________________________________
🔹 Tips & Notes
•	For higher mAP, use more epochs (50–100+) and larger input images.
•	Mixed precision (half=True) reduces VRAM usage.
•	You can switch between yolov8n.pt (Nano), yolov8m.pt (Medium), yolov8l.pt (Large), and yolov8x.pt (Extra-Large) models depending on GPU memory and required accuracy.
•	All results, checkpoints, and predictions are saved to the folder specified in project parameter.
________________________________________
🔹 License
MIT License – free for personal, research, and commercial use.
________________________________________
🔹 Contact
Ch. Sragvi Sai
📧 Email: sragvisai@example.com 


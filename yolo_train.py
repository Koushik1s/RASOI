from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss



from ultralytics import YOLO
import ultralytics.models.yolo
# Load YOLOv12s
model = YOLO(r"yolo12s.pt")





training_params = {
    # === Core Training ===
    'data': r"C:\Users\koush\OneDrive\Desktop\Previous Think\Final Dataset\train data final submit\data.yaml",
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'lr0': 0.0001,
    'lrf': 0.2,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,

    # === Loss Weights ===
    'box': 7.5,
    'cls': 1.0,
    'dfl': 1.5,

    # === Hardware/Setup ===
    'device': "cpu",
    'name': "rasoi_yolov12s_Focal_Loss_2_based_0.746_f",
    'pretrained': True,

    # === Augmentations ===
    'hsv_h': 0.02,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.15,
    'shear': 0.0,
    'perspective': 0.0002,
    'flipud': 0.0,
    'fliplr': 0.5,
    # 'mosaic': 1.0,
    'mixup': 0.1,
    # 'copy_paste': 0.5,   # ✅ enabled copy-paste augmentation
    'erasing': 0.05,

    # === Advanced ===
    'close_mosaic': 15,
    'overlap_mask': True,
    'single_cls': False,
    'save_period': -1,
    'seed': 42,
    'patience': 30,
    'workers': 4,
    'patience': 0,

    # === New Additions ===
    'freeze': 15        # ✅ freeze first 10 layers during initial epochs
    
}

# # Train with custom focal loss
model.train(
    **training_params
    
)


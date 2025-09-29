🍱 Indian Thali Food Detection – YOLOv12 with Correlation Loss
📌 Overview

This project focuses on multi-class food detection and recognition in Indian thali images using YOLOv12.
The dataset combines RASOI dataset and web-scraped food images across 63 dish classes.

Key highlights:
✅ YOLOv12n/s training with transfer learning
✅ Extensive augmentations & layer freezing strategies
✅ Focal Loss for handling imbalance
✅ Custom correlation-matrix-based loss (leveraging food co-occurrence knowledge)
✅ Plate-wise analysis with misclassification reports

📥 Input Format

Model Path: .pt file of YOLOv12

Dataset: Images + YOLO-style annotations

Correlation Matrix: CSV file (63×63) with food relationship scores

📝 Correlation Matrix

Example (scale 0–5):

Class	Idli	Sambar	Rasam	Chicken Curry	Rice
Idli	5	4	2	1	3
Sambar	4	5	3	1	3
Chicken	1	1	1	5	4
🧪 Metrics Generated
Metric	Description
Precision	Correct predictions / Total predictions
Recall	Correct predictions / Total ground truths
F1 Score	Harmonic mean of Precision and Recall
AP	Average Precision for each class
mAP@0.5	Mean of all class AP scores at IoU=0.5
📊 Example Results

Validation Set

mAP@0.5: 0.93

Strong classes: Rice, Salad, Dal, Raita

Weak classes: Rasam, Paneer Curry (confused without paneer)

Test Set

mAP@0.5: 0.82

Improvements after correlation loss in Plates 4, 7, 8 (removed 4 misclassifications)

🚀 How to Run
# Train
yolo train model=yolov12s.pt data=data.yaml epochs=200 batch=16 imgsz=640

# Evaluate
python evaluation.py --model runs/train/weights/best.pt --data datasets/test

# Custom Loss
python train_with_corr_loss.py --correlation loss/correlation_matrix.csv

📦 Dependencies
pip install ultralytics pandas scikit-learn tqdm openpyxl gdown

🔗 Resources

RASOI Dataset

Google Drive – Predictions & Metrics

Correlation Matrix CSV

✅ Example Output
=== Class-wise Metrics ===
Banana: AP=0.74, Precision=0.81, Recall=0.68
Chutney: AP=0.41, Precision=0.35, Recall=0.28
Rice:    AP=0.91, Precision=0.88, Recall=0.87

=== Evaluation Metrics ===
Precision: 0.72
Recall:    0.69
F1 Score:  0.70
mAP@0.5:   0.73 (test) | 0.85 (val)

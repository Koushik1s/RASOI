import os
import json
import cv2

# ==== CONFIG ====
json_file = r"D:\Previous Think\Final Dataset\class_data.json"   # your JSON file
output_dir = "annotated_test_images_2"  # where to save results
os.makedirs(output_dir, exist_ok=True)

# ==== LOAD JSON ====
with open(json_file, "r") as f:
    data = json.load(f)

root_dir = data["root"]
images_data = data["images"]

# ==== VISUALIZE AND SAVE ====
for img_name, detections in images_data.items():
    img_path = os.path.join(root_dir, img_name)

    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        continue

    img = cv2.imread(img_path)

    for det in detections:
        cls, bbox, conf = det
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Label text
        label = f"{cls} {conf:.2f}"

        # Get text size for background rectangle
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3
        )

        # Draw filled rectangle (background for text)
        cv2.rectangle(
            img,
            (x_min, y_min - text_h - baseline - 4),
            (x_min + text_w, y_min),
            (0, 255, 0),  # green background
            -1            # filled
        )

        # Put bold text
        cv2.putText(
            img, label, (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,          # font scale (bigger text)
            (0, 0, 0),    # black text for contrast
            3,            # bold thickness
            cv2.LINE_AA
        )

    # Save annotated image
    save_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_path, img)
    print(f"✅ Saved: {save_path}")

print("🎉 All images annotated and saved in:", output_dir)

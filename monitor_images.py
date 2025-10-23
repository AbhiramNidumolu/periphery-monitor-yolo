import cv2
import numpy as np
from ultralytics import YOLO
import os

# ------------------------------
# CONFIGURATION
# ------------------------------
MODEL_PATH = r"runs/detect/periphery_finetune2/weights/best.pt"
INPUT_IMAGE = r"dataset/images/train/image2_aug13.jpg"   # Change this path to your test image
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Alert thresholds
ALERT_THRESHOLD = 0.5    # Trigger full alert if IoU > 0.1
WARNING_THRESHOLD = 0.1 # Trigger "near zone" warning if IoU > 0.05

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def get_auto_roi(image):
    """Automatically find ROI using edge detection and contour analysis."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the largest contour (machinery area)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Slightly expand and cap the ROI size
    x = max(0, x - 20)
    y = max(0, y - 20)
    w = min(image.shape[1] - x, w + 60)
    h = min(image.shape[0] - y, h + 60)

    # Cap ROI growth relative to image dimensions
    height, width = image.shape[:2]
    max_w, max_h = int(0.7 * width), int(0.5 * height)
    w = min(w, max_w)
    h = min(h, max_h)

    return (x, y, x + w, y + h)

def check_intersection(boxA, boxB):
    """Compute IoU between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# ------------------------------
# MAIN FUNCTION
# ------------------------------
def main():
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    image = cv2.imread(INPUT_IMAGE)
    if image is None:
        print("Error: Could not read image.")
        return

    roi = get_auto_roi(image)
    if roi is None:
        print("Could not find ROI automatically.")
        return

    (rx1, ry1, rx2, ry2) = roi
    height, width = image.shape[:2]

    # Optional: fixed reference ROI blended with auto ROI (for stability)
    fixed_roi = (
        int(0.15 * width), int(0.55 * height),
        int(0.85 * width), int(0.95 * height)
    )
    roi = tuple(int(0.7 * a + 0.3 * b) for a, b in zip(roi, fixed_roi))

    # Run YOLO inference
    results = model.predict(image, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
    classes = results.boxes.cls.cpu().numpy() if results.boxes else []

    alert_triggered = False
    warning_triggered = False

    # Evaluate detections
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        cls_id = int(classes[i])
        if cls_id == 0:  # 'person'
            iou = check_intersection((x1, y1, x2, y2), roi)
            print(f"IoU with ROI: {iou:.3f}")

            if iou > ALERT_THRESHOLD:
                print(f"ALERT: Person detected inside restricted zone (IoU={iou:.2f})")
                alert_triggered = True
                break
            elif iou > WARNING_THRESHOLD:
                print(f"WARNING: Person near restricted zone (IoU={iou:.2f})")
                warning_triggered = True

    # Visualize ROI
    if alert_triggered:
        color = (0, 0, 255)
        label = "ALERT: PERSON IN RESTRICTED ZONE"
    elif warning_triggered:
        color = (0, 165, 255)
        label = "WARNING: PERSON NEAR ZONE"
    else:
        color = (0, 255, 0)
        label = "SAFE ZONE"

    cv2.rectangle(image, (rx1, ry1), (rx2, ry2), color, 3)
    cv2.putText(image, label, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    output_path = os.path.join(OUTPUT_DIR, os.path.basename(INPUT_IMAGE))
    cv2.imwrite(output_path, image)
    print(f"Output saved at: {output_path}")

    # Skip live display (for headless systems)
    print("Display skipped (no GUI support). Check saved output image.")

    if not alert_triggered and not warning_triggered:
        print("Zone clear. No person detected in or near the restricted region.")
    elif warning_triggered:
        print("Person is close to the restricted area, please maintain distance.")
    elif alert_triggered:
        print("Person entered the restricted area! Immediate attention required.")

if __name__ == "__main__":
    main()

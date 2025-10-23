"""
Generate augmented versions of original training images using Albumentations.
Automatically adjusts YOLO bounding boxes.
Author: Abhiram
"""

import os
import cv2
import albumentations as A
import random
import shutil

# === INPUT AND OUTPUT PATHS ===
INPUT_IMG_DIR = "dataset/original/images"
INPUT_LABEL_DIR = "dataset/original/labels"
OUTPUT_IMG_DIR = "dataset/images/train"
OUTPUT_LABEL_DIR = "dataset/labels/train"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# === AUGMENTATION PIPELINE ===
transform = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.RandomRotate90(p=1.0)
    ], p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.MotionBlur(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.4),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=10, p=0.5),
    A.GaussNoise(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# === AUGMENT IMAGES ===
AUG_PER_IMAGE = 20  # Generate 20 augmented images per input

for img_file in os.listdir(INPUT_IMG_DIR):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(INPUT_IMG_DIR, img_file)
    label_path = os.path.join(INPUT_LABEL_DIR, os.path.splitext(img_file)[0] + ".txt")

    # Read image
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    # Read YOLO labels
    with open(label_path, "r") as f:
        lines = f.readlines()

    bboxes, class_labels = [], []
    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])
        bboxes.append([xc, yc, bw, bh])
        class_labels.append(cls)

    # Apply multiple augmentations
    for i in range(AUG_PER_IMAGE):
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = transformed["image"]
        aug_boxes = transformed["bboxes"]
        aug_labels = transformed["class_labels"]

        out_name = f"{os.path.splitext(img_file)[0]}_aug{i}.jpg"
        out_img_path = os.path.join(OUTPUT_IMG_DIR, out_name)
        out_lbl_path = os.path.join(OUTPUT_LABEL_DIR, os.path.splitext(out_name)[0] + ".txt")

        cv2.imwrite(out_img_path, aug_img)

        with open(out_lbl_path, "w") as f:
            for box, lbl in zip(aug_boxes, aug_labels):
                f.write(f"{lbl} {' '.join(map(str, box))}\n")

print("✅ Dataset augmentation complete.")

import os
import cv2
import shutil
import numpy as np
import albumentations as A
from tqdm import tqdm

# Paths
original_images_folder = "/home/hice1/mgulati30/Final Project/data/dataset/images"
original_labels_folder = "/home/hice1/mgulati30/Final Project/data/dataset/labels"
augmented_images_folder = "/home/hice1/mgulati30/Final Project/data/dataset/augmented/images"
augmented_labels_folder = "/home/hice1/mgulati30/Final Project/data/dataset/augmented/labels"
combined_images_folder = "/home/hice1/mgulati30/Final Project/data/dataset/combined/images"
combined_labels_folder = "/home/hice1/mgulati30/Final Project/data/dataset/combined/labels"

os.makedirs(augmented_images_folder, exist_ok=True)
os.makedirs(augmented_labels_folder, exist_ok=True)
os.makedirs(combined_images_folder, exist_ok=True)
os.makedirs(combined_labels_folder, exist_ok=True)

# Class Mapping
class_mapping = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

# Albumentations Augmentation Pipeline
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))


def clamp_bbox(bbox):
    """Clamp YOLO bounding box values to be within the range [0, 1]"""
    x, y, w, h = bbox  # YOLO format: (x_center, y_center, width, height)
    
    x_min = x - w / 2
    x_max = x + w / 2
    y_min = y - h / 2
    y_max = y + h / 2

    # Clamp x_min, x_max, y_min, y_max to [0, 1]
    x_min = np.clip(x_min, 0, 1)
    x_max = np.clip(x_max, 0, 1)
    y_min = np.clip(y_min, 0, 1)
    y_max = np.clip(y_max, 0, 1)

    # Recalculate x, y, w, h from clamped x_min, x_max, y_min, y_max
    x = (x_min + x_max) / 2
    y = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    
    return [x, y, w, h]


def apply_clamp_to_all_bboxes(bboxes):
    """Apply clamping for all bounding boxes"""
    return [clamp_bbox(bbox) for bbox in bboxes]


def augment_class_data(original_images_folder, original_labels_folder, augmented_images_folder, augmented_labels_folder, target_class_index, multiplier=5):
    os.makedirs(augmented_images_folder, exist_ok=True)
    os.makedirs(augmented_labels_folder, exist_ok=True)
    
    images = [f for f in os.listdir(original_images_folder) if f.endswith((".jpg", ".png"))]
    
    for image_file in tqdm(images, desc=f"Augmenting Class {target_class_index}"):
        image_path = os.path.join(original_images_folder, image_file)
        label_path = os.path.join(original_labels_folder, os.path.splitext(image_file)[0] + ".txt")
        image = cv2.imread(image_path)
        
        if not os.path.exists(label_path):
            continue

        bboxes = []
        class_labels = []

        with open(label_path, 'r') as f:
            for line in f:
                class_id, x, y, w, h = map(float, line.split())
                if int(class_id) == target_class_index:
                    bboxes.append([x, y, w, h])
                    class_labels.append(int(class_id))
        
        if not bboxes:
            continue

        for i in range(multiplier):
            try:
                # Apply augmentations to the image and bounding boxes
                augmented = augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']

                #  Apply clamping logic to the bounding boxes
                aug_bboxes = apply_clamp_to_all_bboxes(aug_bboxes)

                if not aug_bboxes:
                    continue  # Skip if no valid bounding boxes left

                # Save augmented image
                aug_image_name = f"{os.path.splitext(image_file)[0]}_aug_{i}.jpg"
                cv2.imwrite(os.path.join(augmented_images_folder, aug_image_name), aug_image)

                # Save YOLO labels
                aug_label_name = f"{os.path.splitext(image_file)[0]}_aug_{i}.txt"
                with open(os.path.join(augmented_labels_folder, aug_label_name), 'w') as f:
                    for bbox, cls in zip(aug_bboxes, class_labels):
                        f.write(f"{cls} {' '.join(map(str, bbox))}\n")
            except Exception as e:
                print(f"Skipping {image_path} due to: {e}")


def copy_files(src_images, src_labels, dest_images, dest_labels):
    """
    Copies images and labels from source to destination.
    """
    image_files = [f for f in os.listdir(src_images) if f.endswith((".jpg", ".png"))]
    for image_file in tqdm(image_files, desc=f"Copying from {src_images}"):
        # Copy image
        src_image_path = os.path.join(src_images, image_file)
        dest_image_path = os.path.join(dest_images, image_file)
        shutil.copy(src_image_path, dest_image_path)

        # Copy label
        label_file = os.path.splitext(image_file)[0] + ".txt"
        src_label_path = os.path.join(src_labels, label_file)
        dest_label_path = os.path.join(dest_labels, label_file)

        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dest_label_path)
        else:
            print(f"Warning: Label file missing for {image_file}")


if __name__ == "__main__":
    augment_class_data(original_images_folder, original_labels_folder, augmented_images_folder, augmented_labels_folder, target_class_index=1, multiplier=4)  # Class 1: without_mask
    augment_class_data(original_images_folder, original_labels_folder, augmented_images_folder, augmented_labels_folder, target_class_index=2, multiplier=5)  # Class 2: mask_weared_incorrect
    
    print("Combining original and augmented datasets...")

    # Copy original data to the combined folder
    copy_files(original_images_folder, original_labels_folder, combined_images_folder, combined_labels_folder)

    # Copy augmented data to the combined folder
    copy_files(augmented_images_folder, augmented_labels_folder, combined_images_folder, combined_labels_folder)

    print("Combined dataset created successfully!")

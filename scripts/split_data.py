import os
import shutil
import random
from tqdm import tqdm

# Define the paths
combined_images_folder = "/home/hice1/mgulati30/Final Project/data/dataset/combined/images"
combined_labels_folder = "/home/hice1/mgulati30/Final Project/data/dataset/combined/labels"
output_split_folder = "/home/hice1/mgulati30/Final Project/data/dataset/split"

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create split directories
os.makedirs(os.path.join(output_split_folder, "train/images"), exist_ok=True)
os.makedirs(os.path.join(output_split_folder, "train/labels"), exist_ok=True)
os.makedirs(os.path.join(output_split_folder, "val/images"), exist_ok=True)
os.makedirs(os.path.join(output_split_folder, "val/labels"), exist_ok=True)
os.makedirs(os.path.join(output_split_folder, "test/images"), exist_ok=True)
os.makedirs(os.path.join(output_split_folder, "test/labels"), exist_ok=True)

def get_image_label_pairs(images_folder, labels_folder):
    """Return image-label pairs where both image and corresponding label exist."""
    image_files = [f for f in os.listdir(images_folder) if f.endswith((".jpg", ".png"))]
    image_label_pairs = []

    for image_file in image_files:
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_folder, label_file)

        if os.path.exists(label_path):
            image_label_pairs.append((image_file, label_file))
        else:
            print(f"Label missing for image: {image_file}")

    return image_label_pairs

def split_data(image_label_pairs, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Shuffle and split data into train, val, and test sets."""
    random.shuffle(image_label_pairs)
    total_images = len(image_label_pairs)

    train_count = int(train_ratio * total_images)
    val_count = int(val_ratio * total_images)
    test_count = total_images - train_count - val_count

    train_data = image_label_pairs[:train_count]
    val_data = image_label_pairs[train_count:train_count + val_count]
    test_data = image_label_pairs[train_count + val_count:]

    print(f"\nSplit Summary:")
    print(f"Total images: {total_images}")
    print(f"Train: {len(train_data)} images ({train_ratio * 100}%)")
    print(f"Validation: {len(val_data)} images ({val_ratio * 100}%)")
    print(f"Test: {len(test_data)} images ({test_ratio * 100}%)\n")

    return train_data, val_data, test_data

def save_split(image_label_pairs, split_path, images_folder, labels_folder):
    """Copy images and labels to the corresponding split folders."""
    for image_file, label_file in tqdm(image_label_pairs, desc=f"Copying data to {split_path}"):
        # Source paths
        image_src = os.path.join(images_folder, image_file)
        label_src = os.path.join(labels_folder, label_file)

        # Destination paths
        image_dst = os.path.join(split_path, "images", image_file)
        label_dst = os.path.join(split_path, "labels", label_file)

        try:
            shutil.copy(image_src, image_dst)
            shutil.copy(label_src, label_dst)
        except Exception as e:
            print(f"Error copying {image_file}: {e}")

if __name__ == "__main__":
    # Step 1: Get all image-label pairs from the combined dataset
    image_label_pairs = get_image_label_pairs(combined_images_folder, combined_labels_folder)

    # Step 2: Split the data
    train_data, val_data, test_data = split_data(image_label_pairs, train_ratio, val_ratio, test_ratio)

    # Step 3: Save each split
    save_split(train_data, os.path.join(output_split_folder, "train"), combined_images_folder, combined_labels_folder)
    save_split(val_data, os.path.join(output_split_folder, "val"), combined_images_folder, combined_labels_folder)
    save_split(test_data, os.path.join(output_split_folder, "test"), combined_images_folder, combined_labels_folder)

    print("\nData splitting completed successfully!")

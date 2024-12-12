import os
from collections import Counter
import matplotlib.pyplot as plt

# Paths
labels_folder = "/home/hice1/mgulati30/Final Project/data/dataset/labels"  #updated YOLO labels
plots_folder = "/home/hice1/mgulati30/Final Project/data/plots"

os.makedirs(plots_folder, exist_ok=True)

def analyze_class_distribution(labels_folder):
    class_counts = Counter()
    bbox_widths, bbox_heights = [], []

    # Traverse through all label files
    for label_file in os.listdir(labels_folder):
        if label_file.endswith(".txt"):
            with open(os.path.join(labels_folder, label_file), "r") as file:
                for line in file:
                    data = line.strip().split()
                    class_id = int(data[0])
                    class_counts[class_id] += 1
                    _, _, bbox_width, bbox_height = map(float, data[1:])
                    bbox_widths.append(bbox_width)
                    bbox_heights.append(bbox_height)

    return class_counts, bbox_widths, bbox_heights

def plot_class_distribution(class_counts):
    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel("Class ID")
    plt.ylabel("Number of Instances")
    plt.title("Updated Class Distribution (YOLO)")
    plt.xticks(list(class_counts.keys()))
    plt.savefig(os.path.join(plots_folder, "updated_class_distribution.png"))
    plt.close()

def plot_bounding_box_statistics(bbox_widths, bbox_heights):
    plt.figure(figsize=(10, 5))
    plt.scatter(bbox_widths, bbox_heights, alpha=0.5)
    plt.xlabel("Bounding Box Width")
    plt.ylabel("Bounding Box Height")
    plt.title("Bounding Box Size Distribution (Updated YOLO)")
    plt.savefig(os.path.join(plots_folder, "updated_bbox_size_distribution.png"))
    plt.close()

# Perform analysis
class_counts, bbox_widths, bbox_heights = analyze_class_distribution(labels_folder)

# Print class distribution
print("\n--- Updated Class Distribution ---")
for class_id, count in class_counts.items():
    print(f"Class {class_id}: {count} instances")

# Print bounding box statistics
print("\n--- Bounding Box Statistics ---")
print(f"Average Width: {sum(bbox_widths) / len(bbox_widths):.4f}")
print(f"Average Height: {sum(bbox_heights) / len(bbox_heights):.4f}")
print(f"Aspect Ratio (width/height): {sum(bw/bh for bw, bh in zip(bbox_widths, bbox_heights)) / len(bbox_widths):.4f}")

# Generate plots
plot_class_distribution(class_counts)
plot_bounding_box_statistics(bbox_widths, bbox_heights)

print("\n--- Analysis and Plots Completed ---")
print(f"Class distribution and bounding box statistics saved in: {plots_folder}")

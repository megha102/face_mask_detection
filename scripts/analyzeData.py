import os
import cv2
from collections import Counter
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt

annotations_folder = "/home/hice1/mgulati30/Final Project/data/dataset/annotations"
labels_folder = "/home/hice1/mgulati30/Final Project/data/dataset/labels"
images_folder = "/home/hice1/mgulati30/Final Project/data/dataset/images"
plots_folder = "/home/hice1/mgulati30/Final Project/plots"
combined_labels_folder = "/home/hice1/mgulati30/Final Project/data/dataset/combined/labels"
plots_folder = "/home/hice1/mgulati30/Final Project/plots"

def analyze_classes_txt(labels_folder):
    class_counts = Counter()
    
    # Traverse through all label files
    for label_file in os.listdir(labels_folder):
        if label_file.endswith(".txt"):
            with open(os.path.join(labels_folder, label_file), 'r') as file:
                for line in file:
                    class_id = int(line.split()[0])  # Extract the class index
                    class_counts[class_id] += 1

    # Display class counts
    print("Class Counts (YOLO):")
    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} instances")
        
    return class_counts


def analyze_classes_xml(annotations_folder):
    class_counts = Counter()

    # Traverse through all XML files
    for xml_file in os.listdir(annotations_folder):
        if xml_file.endswith(".xml"):
            tree = ET.parse(os.path.join(annotations_folder, xml_file))
            root = tree.getroot()

            # Extract class names
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_counts[class_name] += 1

    # Display class counts
    print("Class Counts (Pascal VOC):")
    for class_name, count in class_counts.items():
        print(f"Class '{class_name}': {count} instances")
        
    return class_counts


def analyze_bounding_boxes(labels_folder):
    bbox_widths, bbox_heights, aspect_ratios = [], [], []

    # Traverse through all label files
    for label_file in os.listdir(labels_folder):
        if label_file.endswith(".txt"):
            with open(os.path.join(labels_folder, label_file), 'r') as file:
                for line in file:
                    _, _, _, w, h = map(float, line.split())
                    bbox_widths.append(w)
                    bbox_heights.append(h)
                    aspect_ratios.append(w / h)

    return bbox_widths, bbox_heights



def analyze_image_resolution(images_folder):
    resolutions = []
    aspect_ratios = []

    # Traverse through all image files
    for image_file in os.listdir(images_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image = cv2.imread(os.path.join(images_folder, image_file))
            height, width, _ = image.shape
            resolutions.append((width, height))
            aspect_ratios.append(width / height)

    return resolutions, aspect_ratios



def plot_class_distribution(class_counts, output_file):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel("Class Names")
    plt.ylabel("Number of Instances")
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.savefig(output_file)
    plt.close()
    
def plot_bounding_box_statistics(bbox_widths, bbox_heights, output_file):
    plt.figure(figsize=(10, 5))
    plt.scatter(bbox_widths, bbox_heights, alpha=0.5)
    plt.xlabel("Bounding Box Width")
    plt.ylabel("Bounding Box Height")
    plt.title("Bounding Box Size Distribution")
    plt.savefig(output_file)
    plt.close()
    
def plot_image_resolution_statistics(resolutions, output_file):
    resolution_counts = Counter(resolutions)
    most_common_resolutions = resolution_counts.most_common(5)
    labels = [f"{res[0][0]}x{res[0][1]}" for res in most_common_resolutions]
    counts = [res[1] for res in most_common_resolutions]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts, color='lightgreen')
    plt.xlabel("Resolutions")
    plt.ylabel("Frequency")
    plt.title("Top 5 Image Resolutions")
    plt.savefig(output_file)
    plt.close()
    
    
def data_analysis_pipeline(images_folder, labels_folder, annotations_folder=None, plots_folder="./plots"):
    print("\n--- Analyzing Classes (YOLO)...")
    yolo_class_counts = analyze_classes_txt(labels_folder)
    plot_class_distribution(yolo_class_counts, os.path.join(plots_folder, "class_distribution_yolo.png"))

    if annotations_folder:
        print("\n--- Analyzing Classes (Pascal VOC)...")
        voc_class_counts = analyze_classes_xml(annotations_folder)
        plot_class_distribution(voc_class_counts, os.path.join(plots_folder, "class_distribution_voc.png"))

    print("\n--- Analyzing Bounding Boxes...")
    bbox_widths, bbox_heights = analyze_bounding_boxes(labels_folder)
    plot_bounding_box_statistics(bbox_widths, bbox_heights, os.path.join(plots_folder, "bbox_statistics.png"))

    print("\n--- Analyzing Image Resolutions...")
    resolutions, aspect_ratios = analyze_image_resolution(images_folder)
    plot_image_resolution_statistics(resolutions, os.path.join(plots_folder, "image_resolutions.png"))

    
    
data_analysis_pipeline(images_folder, labels_folder, annotations_folder, plots_folder)
combined_class_counts = analyze_classes_txt(combined_labels_folder)
plot_class_distribution(combined_class_counts, os.path.join(plots_folder, "class_distribution_combined.png"))

print("Class distribution for combined dataset plotted successfully.")
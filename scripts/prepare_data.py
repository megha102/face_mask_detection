import os
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Paths
annotations_folder = "/home/hice1/mgulati30/Final Project/data/dataset/annotations"
images_folder = "/home/hice1/mgulati30/Final Project/data/dataset/images"
labels_folder = "/home/hice1/mgulati30/Final Project/data/dataset/labels"
class_mapping = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

os.makedirs(labels_folder, exist_ok=True)


def convert_xml_to_yolo(xml_file, labels_folder, class_mapping):
    """
    Converts a single XML annotation file in Pascal VOC format to YOLO format and writes the result to a TXT file.

    Args:
        xml_file (str): Path to the XML file.
        labels_folder (str): Path to the folder where YOLO labels will be saved.
        class_mapping (dict): Dictionary mapping class names to YOLO class indices.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract image dimensions
        size = root.find("size")
        image_width = int(size.find("width").text)
        image_height = int(size.find("height").text)

        # Prepare output label file path
        image_filename = root.find("filename").text
        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        label_path = os.path.join(labels_folder, label_filename)

        print(f"Processing: {image_filename}")
        with open(label_path, "w") as label_file:
            for obj in root.findall("object"):
                class_name = obj.find("name").text

                # Skip unrecognized classes
                if class_name not in class_mapping:
                    print(f"  Warning: Unrecognized class '{class_name}' in {xml_file}")
                    continue

                class_index = class_mapping[class_name]

                # Extract and normalize bounding box coordinates
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                x_center = ((xmin + xmax) / 2) / image_width
                y_center = ((ymin + ymax) / 2) / image_height
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height

                # Write YOLO format data
                label_file.write(f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                print(f"  Class: {class_name} (ID: {class_index}), BBox: {x_center:.6f}, {y_center:.6f}, {width:.6f}, {height:.6f}")
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {e}")
    except Exception as e:
        print(f"Unexpected error processing {xml_file}: {e}")
        

def convert_all_xml_to_yolo(annotations_folder, labels_folder, class_mapping):
    """
    Converts all XML annotation files in a folder to YOLO format.

    Args:
        annotations_folder (str): Path to the folder containing XML annotation files.
        labels_folder (str): Path to the folder where YOLO labels will be saved.
        class_mapping (dict): Dictionary mapping class names to YOLO class indices.
    """
    xml_files = [f for f in os.listdir(annotations_folder) if f.endswith(".xml")]
    if not xml_files:
        print("No XML files found in the annotations folder.")
        return

    for xml_file in xml_files:
        xml_path = os.path.join(annotations_folder, xml_file)
        convert_xml_to_yolo(xml_path, labels_folder, class_mapping)

    print("\nConversion completed. YOLO labels saved to:", labels_folder)



def verify_yolo_conversion(xml_folder, yolo_folder, class_mapping, epsilon=1e-5):
    mismatches = []
    missing_yolo_files = []
    unrecognized_classes = []

    # Iterate through XML files
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_folder, xml_file)
        yolo_file = os.path.join(yolo_folder, os.path.splitext(xml_file)[0] + ".txt")

        # Check for YOLO file existence
        if not os.path.exists(yolo_file):
            missing_yolo_files.append(xml_file)
            continue

        # Parse XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        image_width = int(size.find("width").text)
        image_height = int(size.find("height").text)

        # Parse YOLO file
        with open(yolo_file, "r") as yolo:
            yolo_lines = yolo.readlines()

        # Compare bounding boxes between XML and YOLO
        for obj, yolo_line in zip(root.findall("object"), yolo_lines):
            class_name = obj.find("name").text

            if class_name not in class_mapping:
                unrecognized_classes.append((xml_file, class_name))
                continue

            class_index = class_mapping[class_name]
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / image_width
            y_center = ((ymin + ymax) / 2) / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            # Parse YOLO line
            yolo_values = yolo_line.strip().split()
            yolo_class_index = int(yolo_values[0])
            yolo_x_center, yolo_y_center, yolo_width, yolo_height = map(float, yolo_values[1:])

            # Check for mismatches
            if (
                yolo_class_index != class_index or
                abs(yolo_x_center - x_center) > epsilon or
                abs(yolo_y_center - y_center) > epsilon or
                abs(yolo_width - width) > epsilon or
                abs(yolo_height - height) > epsilon
            ):
                mismatches.append({
                    "file": xml_file,
                    "expected": (class_index, x_center, y_center, width, height),
                    "found": (yolo_class_index, yolo_x_center, yolo_y_center, yolo_width, yolo_height)
                })

    # Report Results
    print("\n--- YOLO Conversion Verification Results ---")

    if missing_yolo_files:
        print(f"Missing YOLO files: {len(missing_yolo_files)}")
        for file in missing_yolo_files[:10]:  # Show a few examples
            print(f"  - {file}")
    else:
        print("No missing YOLO files.")

    if unrecognized_classes:
        print(f"Unrecognized classes: {len(unrecognized_classes)}")
        for file, class_name in unrecognized_classes[:10]:  # Show a few examples
            print(f"  - {file}: {class_name}")
    else:
        print("No unrecognized classes.")

    if mismatches:
        print(f"Bounding box mismatches: {len(mismatches)}")
        for mismatch in mismatches[:10]:  # Show a few examples
            print(f"  - File: {mismatch['file']}")
            print(f"    Expected: {mismatch['expected']}")
            print(f"    Found:    {mismatch['found']}")
    else:
        print("No bounding box mismatches.")

    print("\nVerification completed.")



def visualize_yolo_labels(image_folder, yolo_folder, class_mapping, num_images=1, save_dir="/home/hice1/mgulati30/Final Project/plots"):
    colors = ['red', 'blue', 'green', 'yellow']  # Extend as needed
    for idx, image_file in enumerate(os.listdir(image_folder)):
        if not image_file.endswith(('.jpg', '.png')):
            continue
        if idx >= num_images:
            break

        # Load image
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        # Load corresponding YOLO label file
        yolo_file = os.path.join(yolo_folder, os.path.splitext(image_file)[0] + ".txt")
        if not os.path.exists(yolo_file):
            continue

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)

        # Plot YOLO bounding boxes
        with open(yolo_file, "r") as file:
            for line in file:
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
                x1 = int((x_center - bbox_width / 2) * width)
                y1 = int((y_center - bbox_height / 2) * height)
                x2 = int((x_center + bbox_width / 2) * width)
                y2 = int((y_center + bbox_height / 2) * height)

                rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=colors[int(class_id) % len(colors)], facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 10, list(class_mapping.keys())[int(class_id)], color='white', fontsize=12, bbox=dict(facecolor=colors[int(class_id) % len(colors)], edgecolor='black'))

        plt.title(image_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, os.path.splitext(image_file)[0] + "_bboxes.png"))
        plt.show()



convert_all_xml_to_yolo(annotations_folder, labels_folder, class_mapping)
verify_yolo_conversion(annotations_folder,labels_folder,class_mapping)
visualize_yolo_labels(images_folder,labels_folder, class_mapping )

import os
import re

val_path = "/home/hice1/mgulati30/Final\\ Project/yolov5/val.py"

# Define epochs and corresponding weight paths
weights = {
    # 40: "/home/hice1/mgulati30/Final\\ Project/yolov5/runs/train/exp6/weights/best.pt",
    # 100: "/home/hice1/mgulati30/Final\\ Project/yolov5/runs/train/exp9/weights/best.pt",
    140: "/home/hice1/mgulati30/Final\\ Project/yolov5/runs/train/exp13/weights/best.pt",
    200: "/home/hice1/mgulati30/Final\\ Project/yolov5/runs/train/exp11/weights/best.pt",
}

test_results = []

# Loop through epochs and run validation
for epoch, weight_path in weights.items():
    command = f"""
    python {val_path} \
        --weights {weight_path} \
        --data /home/hice1/mgulati30/Final\\ Project/data/dataset/split/data.yaml \
        --conf-thres 0.5 \
        --iou-thres 0.5 \
        --task test \
        --device 0,1
    """
    print(f"Running validation for epoch {epoch} with weights at {weight_path}")
    
    # Capture the output of the validation command
    output = os.popen(command).read()
    print(output)  # Debugging: Print output to verify structure

    # Extract total instances from "all" row
    all_row_match = re.search(r"all\s+\d+\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", output)
    if all_row_match:
        total_instances = int(all_row_match.group(1))  # Extract total instances
        precision = float(all_row_match.group(2))      # Extract precision
        recall = float(all_row_match.group(3))         # Extract recall
        mAP50 = float(all_row_match.group(4))          # Extract mAP@50
        # Calculate correct and incorrect detections
        correct_detections = int(recall * total_instances)
        incorrect_detections = total_instances - correct_detections
    else:
        print("Failed to extract metrics from 'all' row.")
        correct_detections = 0
        incorrect_detections = 0

    # Append results
    test_results.append((epoch, correct_detections, incorrect_detections))

# Print the results table
print("\nEpochs\tCorrect Detections\tIncorrect Detections")
for epoch, correct, incorrect in test_results:
    print(f"{epoch}\t{correct}\t{incorrect}")

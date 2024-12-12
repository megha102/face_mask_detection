Face Mask Detection Using YOLOv5

## Project Overview

This project implements a face mask detection model using the YOLOv5 framework. The objective is to detect face mask usage across three categories: `with_mask`, `without_mask`, and `mask_weared_incorrect`. The dataset was sourced from Kaggle, preprocessed, augmented, and trained using YOLOv5 with hyperparameter modifications.

## Project Structure
```
Final Project/
├── data/
│   ├── dataset/
│   │   ├── annotations/      # Original annotations in XML format
│   │   ├── augmented/        # Augmented images and labels
│   │   ├── combined/         # Combined dataset before splitting
│   │   ├── images/           # Original images
│   │   ├── labels/           # YOLO-format labels
│   │   ├── split/            # Train, validation, and test splits
│   │   ├── plots/            # Data visualization plots
│   └── downloadFromKaggle.ipynb # Kaggle data download notebook
├── plots/                    # Visualized graphs (PR curves, confusion matrices)
├── research_paper/           # Report files
├── runs/                     # Saved YOLOv5 training and validation results
├── scripts/                  # Python scripts for data processing and training
│   ├── analyzeData.py        # Data analysis and visualization
│   ├── augment_data.py       # Data augmentation script
│   ├── dataIntegrityAfterConversion.py # Integrity checks after conversion
│   ├── prepare_data.py       # Dataset preparation
│   ├── split_data.py         # Dataset splitting
│   ├── train_config.py       # Training configuration for YOLOv5
├── yolov5/                   # YOLOv5 framework and scripts
│   ├── models/               # YOLOv5 model configurations
│   ├── data/                 # Hyp files for training
│   ├── train.py              # Modified YOLOv5 training script
│   ├── val.py                # Validation script
├── visualization/            # Code for generating PR curves and graphs
├── env.yml                   # Conda environment file
└── README.md                 # Project documentation
```

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-repo-url>
   cd Final Project
   ```
2. Create and activate the environment:
   ```bash
   conda env create -f env.yml
   conda activate face-mask-detection
   ```
3. Install YOLOv5 requirements:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   pip install -r requirements.txt
   ```

### Dataset Preparation
1. Download the dataset from Kaggle:
   ```bash
   kaggle datasets download -d <dataset-id>
   ```
2. Extract the dataset into `data/dataset/`.
3. Run the preprocessing and augmentation scripts:
   ```bash
   python scripts/prepare_data.py
   python scripts/augment_data.py
   python scripts/split_data.py
   ```

## Training

The training process was conducted using the YOLOv5 framework. For different epochs and GPU configurations, the following command was used:

```bash
python yolov5/train.py \
    --data data/dataset/split/data.yaml \
    --cfg yolov5/models/yolov5m.yaml \
    --hyp yolov5/data/hyps/hyp.scratch-med.yaml \
    --epochs <number_of_epochs> \
    --batch-size 128 \
    --device 0,1,2,3 \
    --workers 16
```

**Training Configurations:**
- **40 Epochs**: `yolov5/runs/train/exp6` (2 GPUs)
- **100 Epochs**: `yolov5/runs/train/exp9` (4 GPUs)
- **200 Epochs**: `yolov5/runs/train/exp11` (4 GPUs)
- **300 Epochs**: `yolov5/runs/train/exp13` (2 GPUs, 4 hours)

## Testing

To validate the trained models on the test set:
```bash
python yolov5/val.py \
    --weights yolov5/runs/train/<exp_folder>/weights/best.pt \
    --data data/dataset/split/data.yaml \
    --conf-thres 0.5 \
    --iou-thres 0.5 \
    --task test \
    --device 0,1
```

### Saved Testing Results
- Results (e.g., precision, recall, mAP) are saved in:
  - `yolov5/runs/val/<exp_folder>`

## Results

### Training Results
The PR curves and metrics for the model trained for 40, 100, 200, and 300 epochs are saved under the respective training directories.

### Testing Results
The model was tested on the split test set. Evaluation metrics include precision, recall, mAP@0.5, and confusion matrices. The results are summarized in the report and the following directories:
- `yolov5/runs/val/`

### Key Observations:
- **Precision-Recall Curves**: Highlighted in the report.
- **Face Mask Detection Table**: Quantitative performance for each epoch is summarized in the report.

## Project Notes

### Challenges Encountered
- Class imbalance: Resolved using data augmentation techniques.
- Training custom scripts: Initially attempted a custom training pipeline but later adapted the YOLOv5 training script for reliability and scalability.
- Validation debugging: Encountered issues during validation, resolved by leveraging YOLOv5’s built-in scripts.

### Improvements
- Used AdamW optimizer and a cyclical learning rate scheduler for enhanced training performance.
- Augmentation techniques improved model generalization.

---

The Yolov5 folder was removed since it was taking too much time. The results are saved in runs - detections. 

import argparse
import math
import os
import random
import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
import yaml
from tqdm import tqdm

from yolov5.utils.general import LOGGER, check_dataset, check_img_size, colorstr, increment_path
from yolov5.utils.metrics import ap_per_class
from yolov5.utils.general import non_max_suppression
from yolov5.utils.loggers import Loggers
from yolov5.utils.torch_utils import EarlyStopping, ModelEMA, select_device
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.dataloaders import create_dataloader
from yolov5.models.yolo import Model

from sklearn.metrics import precision_recall_curve, average_precision_score

import torch
import pandas as pd
import os

import yaml
from collections import Counter

# Load dataset
data_yaml = "/home/hice1/mgulati30/Final Project/data/dataset/split/data.yaml";
with open(data_yaml, 'r') as f:
    data = yaml.safe_load(f)

train_labels = "/home/hice1/mgulati30/Final Project/data/dataset/split/train/labels"  #
val_labels = "/home/hice1/mgulati30/Final Project/data/dataset/split/val/labels"

def count_classes(label_path):
    counts = Counter()
    for label_file in os.listdir(label_path):
        with open(os.path.join(label_path, label_file), 'r') as lf:
            for line in lf:
                class_id = int(line.split()[0])
                counts[class_id] += 1
    return counts

print("Training Class Counts:", count_classes(train_labels))
print("Validation Class Counts:", count_classes(val_labels))


def train(hyp, opt, device):
    results = {
        "epoch": [],
        "box_loss": [],
        "obj_loss": [],
        "class_loss": [],
        "val_box_loss": [],
        "val_obj_loss": [],
        "val_class_loss": [],
        "val_loss": [],
        "mAP_50": [] 
    }

    save_dir, epochs, batch_size, weights, data, cfg, patience, mode = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.data,
        opt.cfg,
        opt.patience,
        opt.mode,
    )
    save_dir.mkdir(parents=True, exist_ok=True)  
    log_path = save_dir / "loss_log.csv"
    predictions_dir = save_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)  # Create directory for saving predictions

    logging.basicConfig(filename=save_dir / 'training.log', level=logging.INFO)

    if not log_path.exists():
        with open(log_path, 'w') as f:
            f.write("epoch,box_loss,obj_loss,class_loss,val_loss\n")

    # Load hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)
    LOGGER.info(colorstr("Hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))

    # Dataset
    data_dict = check_dataset(data)
    train_path, val_path = data_dict["train"], data_dict["val"]
    nc = int(data_dict["nc"])  # Number of classes
    names = data_dict["names"]  # Class names
    LOGGER.info(f"class number from data.yaml: {nc}")
    LOGGER.info(f"Class names from data.yaml: {names}")

    # Model
    model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)
    model.hyp = hyp  # Attach hyperparameters to model

    for k, v in model.named_parameters():
        v.requires_grad = True

    gs = max(int(model.stride.max()), 32)  # Grid size
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # Verify imgsz is gs-multiple

    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, hyp=hyp, augment=True, workers=opt.workers)
    val_loader = create_dataloader(val_path, imgsz, batch_size, gs, hyp=hyp, augment=False, workers=opt.workers)[0]

    optimizer = torch.optim.SGD(model.parameters(), lr=hyp["lr0"], momentum=0.937, weight_decay=hyp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    ema = ModelEMA(model)
    compute_loss = ComputeLoss(model)
    stopper = EarlyStopping(patience=patience)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        mloss = torch.zeros(3, device=device)
        for i, (imgs, targets, paths, _) in enumerate(train_loader):
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets)

            if torch.isnan(loss):
                LOGGER.warning(f"NaN loss encountered. Skipping batch {i}.")
                continue

            loss.backward()
            optimizer.step()
            scheduler.step()
            mloss = (mloss * i + loss_items) / (i + 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_loss_components = torch.zeros(3, device=device)
        predictions_list = []  # To store predictions for all images

        with torch.no_grad():
            for i, (imgs, targets, paths, _) in enumerate(val_loader):
                imgs = imgs.to(device).float() / 255.0  # Normalize
                targets = targets.to(device)

                pred, *_ = model(imgs)
                batch_size = pred.shape[0]
                pred = reshape_yolo_predictions(pred, batch_size)
                loss, loss_items = compute_loss(pred, targets)
                val_loss += loss.item()
                val_loss_components += loss_items

                nms_preds = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

                for idx, single_image_preds in enumerate(nms_preds):
                    image_id = paths[idx].split("/")[-1]
                    if single_image_preds is not None:
                        for pred_box in single_image_preds:
                            x1, y1, x2, y2, conf, class_id = pred_box.tolist()
                            predictions_list.append({
                                'epoch': epoch + 1,
                                'image_id': image_id,
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'confidence': conf,
                                'class_id': class_id
                            })

        val_loss /= (i + 1)
        val_loss_components /= (i + 1)

        LOGGER.info(f"Validation Loss Breakdown - Box: {val_loss_components[0]:.4f}, Obj: {val_loss_components[1]:.4f}, Class: {val_loss_components[2]:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            LOGGER.info(f"Best model saved to {save_dir / 'best_model.pt'} with val_loss {val_loss:.6f}")

        if stopper(epoch, val_loss):
            LOGGER.info(f"Early stopping at epoch {epoch + 1} with best val_loss {best_loss:.6f}")
            break

        results["epoch"].append(epoch + 1)
        results["box_loss"].append(mloss[0].item())
        results["obj_loss"].append(mloss[1].item())
        results["class_loss"].append(mloss[2].item())
        results["val_box_loss"].append(val_loss_components[0].item())
        results["val_obj_loss"].append(val_loss_components[1].item())
        results["val_class_loss"].append(val_loss_components[2].item())
        results["val_loss"].append(val_loss)

        # Save predictions for this epoch
        predictions_df = pd.DataFrame(predictions_list)
        predictions_file = predictions_dir / f"predictions_epoch_{epoch + 1}.csv"
        predictions_df.to_csv(predictions_file, index=False)
        LOGGER.info(f"Saved predictions for epoch {epoch + 1} to {predictions_file}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_dir / "training_results.csv", index=False)
    LOGGER.info(f"Training completed in {(time.time() - t0) / 3600:.2f} hours.")
    torch.save(model.state_dict(), save_dir / "model.pt")
    LOGGER.info(f"Model saved to {save_dir / 'model.pt'}")
    plot_training_results(save_dir / "training_results.csv", save_dir)



def plot_metrics(precision, recall, ap, ap_class, names, mAP, save_path=None):
    """
    Plot Precision-Recall curve for each class and save or display the figure.

    Args:
        precision (list): List of precision values per class.
        recall (list): List of recall values per class.
        ap (torch.Tensor): AP values per class.
        ap_class (list): Class indices.
        names (list): Class names.
        mAP (float): Mean Average Precision at IoU 0.5.
        save_path (str or Path, optional): Path to save the plot. If None, displays the plot.
    """
    plt.figure(figsize=(10, 6))
    for i, c in enumerate(ap_class):
        plt.plot(recall[i], precision[i], label=f"{names[c]} (AP: {ap[i, 0]:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (mAP@0.5: {mAP:.3f})")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"PR curve saved to {save_path}")
    else:
        plt.show()


def reshape_yolo_predictions(pred, batch_size):
    """Reshapes YOLO flattened predictions back into 3 feature maps."""
    try:
        LOGGER.info(f"Shape of pred before reshape: {pred.shape}")
        pred1 = pred[:, :19200, :].reshape(batch_size, 3, 80, 80, 8)
        pred2 = pred[:, 19200:24000, :].reshape(batch_size, 3, 40, 40, 8)
        pred3 = pred[:, 24000:, :].reshape(batch_size, 3, 20, 20, 8)
        return [pred1, pred2, pred3]
    except Exception as e:
        LOGGER.error(f"Error reshaping YOLO predictions: {e}")
        LOGGER.error(f"Shape of pred: {pred.shape}, batch_size: {batch_size}")
        raise e  # Reraise the error to debug it properly




def validate(model, val_loader, batch_size, imgsz, device, save_dir):
    """
    Validates the YOLOv5 model on the validation set and logs the precision, recall, and mAP scores.
    
    Args:
        model_path (str): Path to the saved model checkpoint.
        data_path (str): Path to the data.yaml file.
        batch_size (int): Batch size for validation.
        imgsz (int): Image size for validation.
        device (str): Device to run validation on (CPU, GPU, etc.).
        save_dir (str): Directory where validation results (PR curves) will be saved.
    """
    model.eval()

    # Track results
    all_true_labels = []
    all_pred_scores = []
    all_pred_labels = []

    try:
        # Validation loop
        with torch.no_grad():
            for batch_idx, (imgs, targets, paths, shapes) in enumerate(tqdm(val_loader, desc='Validation')):
                imgs = imgs.to(device).float() / 255.0  # Normalize
                targets = targets.to(device)

                # Forward pass
                preds = model(imgs)
                LOGGER.info(f"Prediction shapes before NMS: {[pred.shape for pred in preds]}")
                preds = non_max_suppression(preds, conf_thres=0.01, iou_thres=0.45, max_det=300)

                for i, pred in enumerate(preds):
                    img_targets = targets[targets[:, 0] == i]  # Get ground-truth targets for the current image
                    if img_targets.size(0) > 0:
                        all_true_labels.extend(img_targets[:, 1])  # Class IDs for ground-truths

                    if pred is not None and pred.size(0) > 0:
                        all_pred_scores.extend(pred[:, 4])  # Confidence scores
                        all_pred_labels.extend(pred[:, 5])  # Predicted class labels
                        
    except Exception as e:
        LOGGER.error(f"Error during validation loop: {e}")
        return

    # Check if there are predictions to calculate metrics
    if not all_true_labels or not all_pred_scores:
        LOGGER.warning("No predictions available to compute metrics.")
        return

    try:
        # Calculate and log precision, recall, and mAP
        calculate_metrics(all_true_labels, all_pred_labels, all_pred_scores, num_classes, class_names, save_dir)
    except Exception as e:
        LOGGER.error(f"Error calculating metrics: {e}")


def calculate_metrics(true_labels, pred_labels, pred_scores, num_classes, class_names, save_dir):
    """
    Calculates precision, recall, F1, and mAP metrics, and saves the precision-recall curves.
    
    Args:
        true_labels (list): List of true class labels.
        pred_labels (list): List of predicted class labels.
        pred_scores (list): List of prediction confidence scores.
        num_classes (int): Number of classes.
        class_names (list): List of class names.
        save_dir (str): Directory to save PR curve plots.
    """
    pr_curves_dir = Path(save_dir) / 'pr_curves'
    pr_curves_dir.mkdir(parents=True, exist_ok=True)

    ap_scores = []
    f1_scores = []

    plt.figure(figsize=(10, 8))

    for class_idx in range(num_classes):
        true_binary = np.array([1 if label == class_idx else 0 for label in true_labels])
        pred_binary_scores = [score for score, label in zip(pred_scores, pred_labels) if label == class_idx]

        if len(pred_binary_scores) == 0 or sum(true_binary) == 0:
            print(f"No data for class {class_names[class_idx]}. Skipping PR curve.")
            ap_scores.append(0.0)
            f1_scores.append(0.0)
            continue

        precision, recall, _ = precision_recall_curve(true_binary, pred_binary_scores)
        ap = average_precision_score(true_binary, pred_binary_scores)
        ap_scores.append(ap)

        f1_scores.append(2 * (precision * recall) / (precision + recall + 1e-6))
        plt.plot(recall, precision, label=f"{class_names[class_idx]} AP={ap:.3f}")

    mean_ap = np.mean(ap_scores)
    mean_f1 = np.nanmean(f1_scores)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Validation)")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(pr_curves_dir / "pr_curve.png")
    plt.close()
    LOGGER.info(f"PR curve saved to {pr_curves_dir}")

    LOGGER.info(f"mAP: {mean_ap:.4f} | Mean F1: {mean_f1:.4f}")



def log_predictions_info(model, val_loader, device, epoch, class_names):
    model.eval()
    class_counts = Counter()
    conf_scores = []

    with torch.no_grad():
        for imgs, targets, paths, _ in val_loader:
            imgs = imgs.to(device).float() / 255.0
            preds = model(imgs)
            preds = non_max_suppression(preds, conf_thres=0.05, iou_thres=0.6, max_det=300)

            for pred in preds:
                if pred is not None and pred.size(0) > 0:
                    class_indices = pred[:, 5].detach().cpu().numpy().astype(int)
                    conf_scores.extend(pred[:, 4].detach().cpu().numpy())  # Collect confidence scores
                    class_counts.update(class_indices)  # Update class counts
                else:
                    LOGGER.warning("No predictions for this image.")

    LOGGER.info(f"Epoch {epoch + 1}: Class distribution in predictions: {class_counts}")
    LOGGER.info(f"Epoch {epoch + 1}: Confidence scores (first 10): {conf_scores[:10]}")


def plot_loss_trends(log_file, save_dir):
    """
    Plot training and validation loss trends over epochs.
    Args:
    - log_file: Path to the loss log file (CSV or TXT).
    - save_dir: Directory to save the loss trend plot.
    """
    # Load loss log (Assumes CSV format: epoch,box_loss,obj_loss,class_loss,val_loss)
    data = np.loadtxt(log_file, delimiter=',', skiprows=1)
    epochs = data[:, 0]
    box_loss = data[:, 1]
    obj_loss = data[:, 2]
    class_loss = data[:, 3]
    val_loss = data[:, 4]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, box_loss, label='Box Loss')
    plt.plot(epochs, obj_loss, label='Object Loss')
    plt.plot(epochs, class_loss, label='Class Loss')
    plt.plot(epochs, val_loss, label='Validation Loss', color='red')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = save_dir / 'loss_trends.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss trend plot saved to {plot_path}")


def evaluate_and_plot_pr_curves(model, val_loader, device, save_dir, epoch, class_names):
    """
    Evaluate and generate Precision-Recall curve for each class and overall mAP.
    Args:
        model: Trained model to evaluate.
        val_loader: Validation dataloader.
        device: Device to run the evaluation on.
        save_dir: Directory to save the PR curve plot.
        epoch: Current epoch for labeling.
        class_names: List of class names for labeling.
    """
    # Variables to store predictions and ground truths
    all_true_labels = []
    all_pred_scores = []
    all_pred_labels = []

    model.eval()  # Set model to evaluation mode

    # Gather predictions and ground truths
    with torch.no_grad():
        for imgs, targets, _, _ in val_loader:
            imgs = imgs.to(device).float() / 255.0  # Normalize images
            targets = targets.to(device)

            preds = model(imgs)  # Get model predictions
            preds = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.45, max_det=300)

            for i, pred in enumerate(preds):
                img_targets = targets[targets[:, 0] == i]  # Ground truths for the current image

                if img_targets.size(0) > 0:
                    all_true_labels.extend(img_targets[:, 1].cpu().numpy())  # Class IDs for ground truths

                if pred is not None and pred.size(0) > 0:
                    LOGGER.info(f"Predictions for image {i}: {pred.shape[0]} objects detected.")
                else:
                    LOGGER.warning(f"No predictions for image {i}.")

                    all_pred_scores.extend(pred[:, 4].cpu().numpy())  # Confidence scores
                    all_pred_labels.extend(pred[:, 5].cpu().numpy())  # Predicted class IDs

    # Calculate Precision-Recall curves
    num_classes = len(class_names)
    ap_scores = []
    plt.figure(figsize=(10, 8))

    for class_idx in range(num_classes):
        # Filter predictions for the current class
        true_binary = np.array([1 if label == class_idx else 0 for label in all_true_labels])
        pred_binary_scores = [
            score for score, label in zip(all_pred_scores, all_pred_labels) if label == class_idx
        ]

        if len(pred_binary_scores) == 0 or sum(true_binary) == 0:
            print(f"No data for class {class_names[class_idx]}. Skipping PR curve.")
            ap_scores.append(0.0)
            continue

        precision, recall, _ = precision_recall_curve(true_binary, pred_binary_scores)
        ap = average_precision_score(true_binary, pred_binary_scores)
        ap_scores.append(ap)

        # Plot the PR curve for the class
        plt.plot(recall, precision, label=f"{class_names[class_idx]} {ap:.3f}")

    # Calculate mean Average Precision (mAP)
    mean_ap = np.mean(ap_scores)
    plt.plot([0, 1], [0, 0], "k--", label=f"all classes mAP@0.5={mean_ap:.3f}")

    # Formatting the plot
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Epoch {epoch})")
    plt.legend(loc="lower left")
    plt.grid(True)

    # Save the plot
    pr_curve_path = save_dir / f"pr_curve_epoch_{epoch}.png"
    plt.savefig(pr_curve_path)
    plt.close()
    print(f"PR curve saved to {pr_curve_path}")

    return mean_ap


def plot_training_results(csv_path, save_dir=None):
    # Load results from CSV
    results_df = pd.read_csv(csv_path)

    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["epoch"], results_df["box_loss"], label="Training Box Loss")
    plt.plot(results_df["epoch"], results_df["obj_loss"], label="Training Objectness Loss")
    plt.plot(results_df["epoch"], results_df["class_loss"], label="Training Class Loss")
    plt.plot(results_df["epoch"], results_df["val_box_loss"], label="Validation Box Loss")
    plt.plot(results_df["epoch"], results_df["val_obj_loss"], label="Validation Objectness Loss")
    plt.plot(results_df["epoch"], results_df["val_class_loss"], label="Validation Class Loss")
    plt.plot(results_df["epoch"], results_df["val_loss"], label="Validation Total Loss", linestyle="--", color="red")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save or show the plot
    if save_dir:
        plt.savefig(save_dir / "training_validation_loss_plot.png")
        print(f"Plot saved to {save_dir / 'training_validation_loss_plot.png'}")
    else:
        plt.show()




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Model config path")
    parser.add_argument("--data", type=str, required=True, help="Dataset config path")
    parser.add_argument("--hyp", type=str, required=True, help="Hyperparameters path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="mps", help="Training device")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--save-dir", type=str, default="./runs/train", help="Save directory")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument('--weights', type=str, default='yolov5m.pt', help='Initial weights path')
    parser.add_argument("--mode", type=str, choices=["default", "customized"], default="default", help="Training mode")

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    device = select_device(opt.device)
    with open(opt.hyp, "r") as f:
        hyp = yaml.safe_load(f)
    train(hyp, opt, device)
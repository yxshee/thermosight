# API Reference

This document describes the main modules and functions in ThermoSight.

---

## `src/data/make_dataset.py`

- **build_dataloaders(input_dir, output_dir, img_size, batch_size, split_ratio=0.8, num_workers=4)**
  - Prepares processed datasets and returns PyTorch DataLoaders.

---

## `src/models/vit_model.py`

- **ViT**  
  Vision Transformer model for image classification.

---

## `src/models/train.py`

- **train(args)**
  - Trains the ViT model and logs metrics.

---

## `src/inference/predict.py`

- **predict_image(image_path, model_path, img_size=460, patch_size=8, device=None)**
  - Runs inference on a single image and returns class probabilities.

---

## `src/utils/visualize.py`

- **plot_metrics(writer, train_loss, test_acc, epoch)**
- **plot_confusion_matrix(writer, cm, epoch, class_names=None)**

---

## `src/evaluate/metrics.py`

- **compute_confusion_matrix(true_labels, pred_labels, num_classes)**
- **per_class_accuracy(cm)**

---

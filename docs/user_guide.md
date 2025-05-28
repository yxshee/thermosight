# User Guide

Welcome to the ThermoSight User Guide!  
This document will help you get started with installation, data preparation, training, and inference.

## 1. Installation

```bash
git clone https://github.com/yourusername/thermosight.git
cd thermosight
pip install -r requirements.txt
```

## 2. Data Preparation

- Place your raw microscope images in `data/raw/`, organized by class.
- Run:

```bash
python src/data/make_dataset.py --input_dir data/raw --output_dir data/processed
```

## 3. Training

```bash
python src/models/train.py --input_dir data/raw --output_dir data/processed
```

## 4. Monitoring

- Launch TensorBoard:

```bash
tensorboard --logdir outputs/logs
```

## 5. Inference

```bash
python src/inference/predict.py path/to/image.jpg --model models/best_model.pth
```

## 6. Visualization

- Use the provided notebooks in `notebooks/` for EDA and results visualization.

---

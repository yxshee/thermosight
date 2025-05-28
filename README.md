<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status">
</p>

<h1 align="center">🔥 ThermoSight 🔥</h1>
<h3 align="center"> Vision Transformer for Thermal Image Classification </h3>

<p align="center">
  <img src="data/assets/sonik" alt="ThermoSight Banner" width="50%"/>
</p>

---

## 🌟 Overview

**ThermoSight** is your smart microscope companion for **thermal image classification**.  
Built on a Vision Transformer (ViT) backbone, it streamlines the journey from raw thermal images to actionable temperature class predictions.

> "Turning invisible heat into visible insights!" 🌡️

---

## 🏗️ Problem Statement

When a building catches fire, the extreme heat can seriously damage the materials it’s made from—especially concrete and cement. These materials react differently at different temperatures, so the damage can vary throughout the building depending on how hot it got in each area.

Traditionally, experts collect tiny pieces of concrete from different parts of the structure and take extremely detailed images of the damaged areas using powerful microscopes. By studying these samples and images, they estimate how much heat each part of the building was exposed to and decide whether the structure is safe, needs repairs, or should be demolished. However, this process is slow and labor-intensive.

**ThermoSight** solves this by using AI to analyze high-resolution microscope images of fire-damaged concrete. Our model estimates the temperature each part of the material was exposed to, enabling rapid assessment of structural safety and guiding recovery actions—whether that means repairs or demolition. This approach speeds up post-fire recovery and safety checks, making the process faster and more efficient.

---

## 🚀 Features

### 🧠 Intelligent Classification

| Feature | Description | Emoji |
|---------|-------------|-------|
| **ViT Backbone** | State-of-the-art Vision Transformer for image classification | 🤖 |
| **Flexible Data Pipeline** | Raw-to-processed splits, easy EDA | 🔄 |
| **TensorBoard Logging** | Metrics, confusion matrices, and more | 📊 |
| **Easy Inference** | Predict on new images with a single command | ⚡ |
| **Creative Visualizations** | EDA and results at your fingertips | 🎨 |

---

## 🛠️ Tech Stack

![PyTorch](https://img.shields.io/badge/-PyTorch-ee4c2c?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![TensorBoard](https://img.shields.io/badge/-TensorBoard-FF6F00?logo=tensorboard&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?logo=matplotlib&logoColor=white)

---

## ⚡ Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/thermosight.git
cd thermosight

# Install dependencies
pip install -r requirements.txt
```

**Prepare Data**  
Organize your raw images by class in `data/raw/`, then run:
```bash
python src/data/make_dataset.py --input_dir data/raw --output_dir data/processed
```

**Train the Model**
```bash
python src/models/train.py --input_dir data/raw --output_dir data/processed
```

**Monitor Training**
```bash
tensorboard --logdir outputs/logs
```

**Run Inference**
```bash
python src/inference/predict.py path/to/image.jpg --model models/best_model.pth
```

---

## 📂 Directory Structure

```
thermosight/
│
├── data/
│   ├── raw/         # Raw microscope images (by class)
│   └── processed/   # Preprocessed train/test splits
│
├── models/          # Saved model checkpoints
├── notebooks/       # Jupyter/VSCode notebooks (EDA, training, inference)
├── src/             # Source code (data, models, utils, inference)
├── outputs/         # Logs, TensorBoard runs
├── requirements.txt
└── README.md
```

---

## 🧪 Notebooks

- **01_exploratory_data_analysis.ipynb**: Visualize class distributions and sample images.
- **02_training_pipeline_experiment.ipynb**: Run and log training experiments.
- **03_inference_demo.ipynb**: Predict and visualize results on new images.

---

## 📈 Results

- **Accuracy**: 95%
- **F1 Score**: 0.94
- **Inference Time**: 200ms/image
- **Confusion Matrix**: Visualized in TensorBoard
- **Class Distribution**: Balanced across temperature classes
- **Sample Predictions**: Visualized in Jupyter Notebook

---

<p align="center">
  <img src="data/assets/04.jpeg" width="355" height="355"  alt="Class Distribution"/>
  <img src="data/assets/03.jpeg" width="355" height="355" alt="Confusion Matrix"/>
  <img src="data/assets/01.jpeg" width="710"  alt="Result Graph"/>
</p>

---



## 📚 Documentation

- **User Guide**: [docs/user_guide.md](docs/user_guide.md)
- **API Reference**: [docs/api_reference.md](docs/api_reference.md)
- **Troubleshooting**: [docs/troubleshooting.md](docs/troubleshooting.md)
- **Contributing Guide**: [docs/contributing.md](docs/contributing.md)

---

## 👥 Meet the Author

<table>
  <tr align="center">
    <td><a href="https://github.com/yxshee"><img src="https://avatars.githubusercontent.com/yxshee" width="100px"><br/>Yash Dogra</a></td>
     <td><a href="https://github.com/raghav-215"><img src="https://avatars.githubusercontent.com/raghav-215" width="100px"><br/>Raghav Sharma</a></td>
    <td><a href="https://github.com/DigvijaySidhu07"><img src="https://avatars.githubusercontent.com/DigvijaySidhu07" width="100px"><br/>Digvijay Sidhu</a></td>
    
  </tr>
</table>

---

## 📜 License

```text
MIT License

Copyright (c) 2025 Yash Dogra

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

<div align="center">
  Made with ❤️ | 🔥 Happy Classifying
</div>

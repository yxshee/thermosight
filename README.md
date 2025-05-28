<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status">
</p>

<h1 align="center">🔥 ThermoSight 🔥</h1>
<h3 align="center"> Vision Transformer for Thermal Image Classification </h3>

<p align="center">
  <img src="data/assets/thermosight-banner.png" alt="ThermoSight Banner" width="700"/>
</p>

---

## 🌟 Overview

<p align="center">
  <img src="https://img.shields.io/badge/AI%20Powered-Yes-blue?style=flat-square&logo=python" alt="AI Powered">
  <img src="https://img.shields.io/badge/🔥-Thermal%20Vision-orange?style=flat-square">
  <img src="https://img.shields.io/badge/ViT-Transformer-yellow?style=flat-square">
</p>

**ThermoSight** is your smart microscope companion for **thermal image classification**.<br>
Built on a Vision Transformer (ViT) backbone, it transforms raw thermal images into actionable temperature class predictions.

<blockquote>
  <b>“Turning invisible heat into visible insights!”</b> <span style="font-size:1.2em;">🌡️</span>
</blockquote>

---

## 🚀 Features

<div align="center">

| 🤖 Feature             | Description                                      | Emoji |
|-----------------------|--------------------------------------------------|-------|
| **ViT Backbone**      | State-of-the-art Vision Transformer for images   | 🧬    |
| **Flexible Pipeline** | Raw-to-processed splits, easy EDA                | 🔄    |
| **TensorBoard Logs**  | Metrics, confusion matrices, and more            | 📊    |
| **Easy Inference**    | Predict on new images with a single command      | ⚡    |
| **Visualizations**    | EDA and results at your fingertips               | 🎨    |

</div>

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/-PyTorch-ee4c2c?logo=pytorch&logoColor=white" height="24">
  <img src="https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white" height="24">
  <img src="https://img.shields.io/badge/-TensorBoard-FF6F00?logo=tensorboard&logoColor=white" height="24">
  <img src="https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white" height="24">
  <img src="https://img.shields.io/badge/-Matplotlib-11557c?logo=matplotlib&logoColor=white" height="24">
</p>

---

## ⚡ Quick Start

<div align="center">
  <img src="https://img.icons8.com/color/48/000000/rocket--v2.png" width="40"/>
</div>

```bash
# 🚀 Clone the repo
git clone https://github.com/yourusername/thermosight.git
cd thermosight

# 📦 Install dependencies
pip install -r requirements.txt
```

**🗂️ Prepare Data**  
Organize your raw images by class in `data/raw/`, then run:
```bash
python src/data/make_dataset.py --input_dir data/raw --output_dir data/processed
```

**🏋️ Train the Model**
```bash
python src/models/train.py --input_dir data/raw --output_dir data/processed
```

**📈 Monitor Training**
```bash
tensorboard --logdir outputs/logs
```

**🔮 Run Inference**
```bash
python src/inference/predict.py path/to/image.jpg --model models/best_model.pth
```

---

## 📂 Directory Structure

<details>
<summary>Click to expand</summary>

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
</details>

---

## 🧪 Notebooks

- <img src="https://img.icons8.com/fluency/24/000000/graph.png"/> **01_exploratory_data_analysis.ipynb**: Visualize class distributions and sample images.
- <img src="https://img.icons8.com/fluency/24/000000/experimental.png"/> **02_training_pipeline_experiment.ipynb**: Run and log training experiments.
- <img src="https://img.icons8.com/fluency/24/000000/ai.png"/> **03_inference_demo.ipynb**: Predict and visualize results on new images.

---

## 📈 Results

<div align="center">
  <img src="https://img.icons8.com/color/48/000000/ok--v2.png" width="32"/>
  <b>Accuracy:</b> <span style="color:green">95%</span> &nbsp; | &nbsp;
  <b>F1 Score:</b> <span style="color:green">0.94</span> &nbsp; | &nbsp;
  <b>Inference Time:</b> <span style="color:blue">200ms/image</span>
</div>

- **Confusion Matrix:** Visualized in TensorBoard
- **Class Distribution:** Balanced across temperature classes
- **Sample Predictions:** Visualized in Jupyter Notebook

---

<p align="center">
  <img src="data/assets/04.jpeg" width="355" height="355"  alt="Class Distribution"/>
  <img src="data/assets/03.jpeg" width="355" height="355" alt="Confusion Matrix"/>
  <img src="data/assets/01.jpeg" width="710"  alt="Result Graph"/>
</p>

---

## 🏗️ Problem Statement

<div align="center">
  <img src="https://img.icons8.com/color/48/000000/fire-element--v2.png" width="40"/>
</div>

When a building catches fire, the extreme heat can seriously damage the materials it’s made from—especially concrete and cement. These materials react differently at different temperatures, so the damage can vary throughout the building depending on how hot it got in each area.

Traditionally, experts collect tiny pieces of concrete from different parts of the structure and take extremely detailed images of the damaged areas using powerful microscopes. By studying these samples and images, they estimate how much heat each part of the building was exposed to and decide whether the structure is safe, needs repairs, or should be demolished. However, this process is slow and labor-intensive.

**ThermoSight** solves this by using AI to analyze high-resolution microscope images of fire-damaged concrete. Our model estimates the temperature each part of the material was exposed to, enabling rapid assessment of structural safety and guiding recovery actions—whether that means repairs or demolition. This approach speeds up post-fire recovery and safety checks, making the process faster and more efficient.

---

## 📚 Documentation

- <img src="https://img.icons8.com/ios-filled/20/000000/book.png"/> **User Guide**: [docs/user_guide.md](docs/user_guide.md)
- <img src="https://img.icons8.com/ios-filled/20/000000/api.png"/> **API Reference**: [docs/api_reference.md](docs/api_reference.md)
- <img src="https://img.icons8.com/ios-filled/20/000000/help.png"/> **Troubleshooting**: [docs/troubleshooting.md](docs/troubleshooting.md)
- <img src="https://img.icons8.com/ios-filled/20/000000/git.png"/> **Contributing Guide**: [docs/contributing.md](docs/contributing.md)

---

## 👥 Meet the Author

<table>
  <tr align="center">
    <td>
      <a href="https://github.com/yxshee">
        <img src="https://avatars.githubusercontent.com/yxshee" width="100px"><br/>
        <b>Yash Dogra</b>
      </a>
    </td>
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
  <img src="https://img.icons8.com/color/48/000000/like--v2.png" width="32"/>
  <b>Made with ❤️ by Yash Dogra | 🔥 Happy Classifying!</b>
</div>

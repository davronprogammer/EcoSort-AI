# EcoSort AI - Intelligent Waste Classification

An AI-powered image classification system that identifies waste categories using Deep Learning and Transfer Learning.

EcoSort AI uses a fine-tuned **EfficientNet-B0** model pretrained on ImageNet to classify uploaded images into 12 different waste categories. The project includes dataset analysis, image preprocessing, model training, evaluation, and a Streamlit web application.

---

sreamlite - https://ecosort-ai-5uyomnqpsjztrwdjmcvvyy.streamlit.app/ 

## Project Overview

Waste classification is an important part of modern recycling and waste management systems. Manually identifying different waste materials can be time-consuming and inefficient.

EcoSort AI explores how computer vision can automate this process by recognizing waste categories from images.

**Main objectives:**
- Build a multi-class image classification system.
- Apply Transfer Learning using ImageNet pretrained models.
- Fine-tune EfficientNet-B0 for waste classification.
- Evaluate model performance using classification metrics.
- Compare different CNN architectures.
- Develop an interactive web application using Streamlit.

## Supported Categories

The model recognizes 12 categories:

| # | Category | Images |
|---:|---|---:|
| 1 | Battery | 945 |
| 2 | Biological | 985 |
| 3 | Brown Glass | 607 |
| 4 | Cardboard | 891 |
| 5 | Clothes | 5,325 |
| 6 | Green Glass | 629 |
| 7 | Metal | 769 |
| 8 | Paper | 1,050 |
| 9 | Plastic | 865 |
| 10 | Shoes | 1,977 |
| 11 | Trash | 697 |
| 12 | White Glass | 775 |
| | **Total** | **15,515** |

## Dataset

**Source:** [Garbage Classification — Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

- Total images: 15,515
- Number of classes: 12
- Corrupted images detected: 0

### Dataset Distribution

The dataset is imbalanced. For example, Clothes contains 5,325 images, while Brown Glass contains only 607 images.

To address this issue, the project uses:
- Stratified data splitting.
- Weighted Cross Entropy Loss.
- Macro F1-score for evaluation.

## Data Preparation

The dataset was divided into three subsets:

| Subset | Images | Percentage |
|---|---:|---:|
| Training | 12,412 | 80% |
| Validation | 1,551 | 10% |
| Testing | 1,552 | 10% |
| **Total** | **15,515** | **100%** |

### Image Preprocessing

All images are converted to RGB and resized to 224 × 224 pixels.

**Training augmentation:**
- Random Resized Crop
- Random Horizontal Flip
- Random Rotation
- Color Jitter
- ImageNet Normalization

**Validation and testing:**
- Resize
- Center Crop
- ImageNet Normalization

## Model Architecture

### EfficientNet-B0

The main architecture is EfficientNet-B0, pretrained on ImageNet.

Instead of training a CNN from scratch, Transfer Learning is used to reuse pretrained visual features.

The original classification layer is replaced with a new fully connected layer containing 12 output neurons.

### Training Strategy

Training consists of two stages:

**Stage 1 — Feature Extraction**
- Freeze the pretrained backbone.
- Train the new classification layer.
- Learning rate: 0.001.
- Epochs: 5.

**Stage 2 — Fine-Tuning**
- Unfreeze the last three feature blocks.
- Fine-tune the backbone and classifier.
- Backbone learning rate: 0.00001.
- Classifier learning rate: 0.0001.
- Epochs: 10.

**Optimization:**
- Optimizer: AdamW
- Loss function: Weighted Cross Entropy Loss
- Learning rate scheduler: ReduceLROnPlateau
- Batch size: 32
- Random seed: 42

## Training Results

The EfficientNet-B0 training process showed stable convergence.

| Metric | Approximate Result |
|---|---:|
| Training Accuracy | 95.6% |
| Validation Accuracy | 95.3% |
| Final Validation Loss | 0.18 |

These values are approximate readings from the training history visualization.

The final test metrics will be reported separately after evaluation on the held-out test set.

### Evaluation Metrics

The project uses:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Confusion Matrix

## Model Comparison

EfficientNet-B0 is the primary model. ResNet50 is included as a second architecture for comparison.

The comparison uses the same training, validation, and test splits.

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| EfficientNet-B0 | To be added | To be added | To be added |
| ResNet50 | To be added | To be added | To be added |

*Comparison metrics should be updated using actual test results.*

## Streamlit Web Application

EcoSort AI includes an interactive web interface built with Streamlit.

**Application features:**
- Upload JPG, JPEG, PNG, or WEBP images.
- Classify uploaded waste images.
- Display predicted category.
- Show prediction confidence.
- Display the Top-3 predictions.

### Application Preview

Add your application screenshot here:

```text
![EcoSort AI Application](results/app_preview.png)
```

## Project Structure

```text
EcoSort-AI/
│
├── notebooks/
│   └── EcoSort_AI.ipynb
│
├── results/
│   ├── classes_of_images.jpg
│   ├── model_comparison.jpg
│   ├── training_history.jpg
│   └── confusion_matrix.png
│
├── app.py
├── class_names.json
├── ecosort_efficientnet_b0.pth
├── requirements.txt
└── README.md
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/davronprogrammer/EcoSort-AI.git

cd EcoSort-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit in your browser.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep Learning framework |
| Torchvision | Pretrained models and image transformations |
| EfficientNet-B0 | Main classification architecture |
| ResNet50 | Model comparison |
| NumPy | Numerical operations |
| Pandas | Dataset analysis |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Evaluation metrics |
| Streamlit | Interactive web application |
| Google Colab | Model training with GPU |

## Limitations

- The dataset has class imbalance.
- Some images may contain incorrect or ambiguous labels.
- Model predictions depend on image quality, lighting, and object visibility.
- Prediction confidence is not a guarantee of correctness.
- The model classifies images into dataset categories; it does not determine whether an object is recyclable under local recycling regulations.

## Future Improvements

- Implement Grad-CAM for model explainability.
- Improve performance on minority classes.
- Compare additional architectures such as MobileNetV3 and ResNet50.
- Add object detection for images containing multiple waste objects.
- Deploy the application publicly.
- Explore real-time camera-based waste classification.

## Author

**Davron Normamatov**

AI / ML Engineering Student

GitHub: [@davronprogrammer](https://github.com/davronprogrammer)

---

## Acknowledgements

Dataset: [Garbage Classification — Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

Pretrained architecture: [EfficientNet — Torchvision](https://pytorch.org/vision/stable/models/efficientnet.html)

Web framework: [Streamlit](https://streamlit.io/)

---

**EcoSort AI — Making Waste Classification Smarter with Artificial Intelligence.**
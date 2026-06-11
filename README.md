# PCA + ANN Face Recognition System

A professional implementation of a **Face Recognition System using Principal Component Analysis (PCA) and Artificial Neural Network (ANN)** for dimensionality reduction and facial classification.

This project implements the **Eigenfaces approach** using PCA for feature extraction and an **ANN classifier trained using backpropagation** for face recognition.

The system also supports **imposter detection**, identifying unknown faces as **“Not Enrolled”**.

---

## Project Overview

Traditional face recognition systems struggle with high-dimensional image data. This project solves the problem using:

* **PCA (Principal Component Analysis)** → Reduces dimensionality and extracts important facial features.
* **Eigenfaces** → Represents dominant facial characteristics.
* **ANN (Artificial Neural Network)** → Classifies faces using extracted PCA signatures.

The model was evaluated using:

* **60% training data**
* **40% testing data**
* **Accuracy vs k-value analysis**
* **Imposter (Unknown Face) Detection**

---

## Features

✔ Face image preprocessing
✔ Mean face generation
✔ Mean-zero transformation
✔ Surrogate covariance matrix implementation
✔ Eigenvalue & eigenvector decomposition using SVD
✔ Top-k feature vector selection
✔ Eigenfaces generation
✔ Face signature creation
✔ ANN-based face classification
✔ Accuracy vs k-value experiment
✔ Imposter detection (**Not Enrolled**)
✔ Professional documentation and modular architecture

---

## Tech Stack

### Programming Language

* Python

### Libraries Used

* NumPy
* SciPy
* OpenCV
* Matplotlib
* Scikit-learn

### Concepts Used

* Principal Component Analysis (PCA)
* Eigenfaces
* Singular Value Decomposition (SVD)
* Artificial Neural Network (ANN)
* Backpropagation
* Face Recognition
* Feature Extraction

---

## Project Architecture

```text
Dataset Images
       ↓
Image Preprocessing
       ↓
Mean Face Calculation
       ↓
Mean-Zero Transformation
       ↓
Covariance Matrix
       ↓
Eigenvalue & Eigenvector Decomposition
       ↓
Top-k Principal Components
       ↓
Eigenfaces Generation
       ↓
Face Signatures
       ↓
ANN Training
       ↓
Prediction / Recognition
       ↓
Unknown Face Detection
```

## Dataset Structure

```text
dataset/
│── faces/
│   ├── Aamir/
│   ├── Ajay/
│   ├── Akshay/
│   ├── Alia/
│   ├── Amitabh/
│   ├── Deepika/
│   ├── Disha/
│   ├── Farhan/
│   └── Ileana/
│
└── imposters/
```

### Dataset Details

* **9 enrolled individuals**
* **450 total face images**
* Images resized to **100 × 100**
* **10 imposter images** used for unknown face testing

---

## Folder Structure

```text
face-recognition-pca-ann/
│── dataset/
│── models/
│── notebooks/
│── outputs/
│   ├── eigenfaces/
│   ├── graphs/
│   └── predictions/
│
│── src/
│   ├── data_loader.py
│   ├── pca.py
│   ├── ann_model.py
│   ├── train.py
│   ├── test.py
│   ├── evaluate.py
│   └── utils.py
│
│── main.py
│── requirements.txt
│── README.md
│── .gitignore
```

## PCA Workflow

### 1. Face Database Generation

All face images are converted into column vectors and stored in a face database matrix.

### 2. Mean Face Calculation

The mean of all face images is computed.

### 3. Mean-Zero Transformation

The mean face is subtracted from each image.

### 4. Covariance Matrix Computation

A surrogate covariance matrix is generated to reduce computational complexity.

### 5. Eigen Decomposition

Eigenvalues and eigenvectors are computed using **Singular Value Decomposition (SVD)**.

### 6. Top-k Feature Selection

The most important eigenvectors are selected based on descending eigenvalues.

### 7. Eigenfaces Generation

Principal facial features are extracted.

### 8. Face Signature Creation

Each face is projected into eigenspace to generate feature signatures.

### 9. ANN Classification

The ANN classifier learns face signatures and predicts identities.

---

## ANN Model Configuration

```python
Hidden Layers: (128, 64)
Activation Function: ReLU
Optimizer: Adam
Learning Rate: Adaptive
Max Iterations: 1000
Early Stopping: Enabled
```

---

## Model Results

### Best Model Accuracy

| Metric       | Value  |
| ------------ | ------ |
| Best k Value | 75     |
| Accuracy     | 60.00% |

---

## Accuracy vs k Experiment

| k Value | Accuracy   |
| ------- | ---------- |
| 10      | 28.89%     |
| 20      | 25.56%     |
| 30      | 52.22%     |
| 50      | 50.00%     |
| 75      | **60.00%** |
| 100     | 48.33%     |
| 125     | 46.11%     |
| 150     | 58.33%     |
| 200     | 40.00%     |

Best performance was achieved at:

```text
k = 75
```

---

## Imposter Detection Results

The system was tested with **10 unknown face images**.

### Result

```text
10 / 10
Not Enrolled
```

The confidence threshold mechanism successfully rejected unknown faces.

Example Output:

```text
img1.jpeg → Not Enrolled
img2.jpeg → Not Enrolled
img3.jpeg → Not Enrolled
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/amritalakshmim/face-recognition-pca-ann.git
```

Navigate to project folder:

```bash
cd face-recognition-pca-ann
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python main.py
```

---

## Output

The project generates:

* Eigenfaces visualization
* Accuracy vs k graph
* Face recognition predictions
* Unknown face detection results

Saved inside:

```text
outputs/
```

---

## Future Improvements

* Real-time face recognition using webcam
* Flask web application deployment
* Deep learning-based face recognition
* Better unknown face threshold tuning
* Larger dataset training

---

## Repository

GitHub Repository:

https://github.com/amritalakshmim/face-recognition-pca-ann

---

## Developed by

**Amritalakshmi M.**

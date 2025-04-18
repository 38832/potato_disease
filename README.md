
# 🌱 Potato Disease Classification Project

Welcome to the **Potato Disease Classification** project! This repository contains a deep learning model designed to identify diseases in potato crops using image classification. Built with TensorFlow and Keras, the model classifies potato leaf images into three categories: *Early Blight*, *Late Blight*, and *Healthy*. The goal is to support farmers and agricultural researchers in early disease detection for healthier crops and sustainable farming.

---

## 🚀 Project Overview

The model is trained on the **PlantVillage** dataset, featuring images of potato leaves affected by *Early Blight*, *Late Blight*, or no disease (*Healthy*). Using a Convolutional Neural Network (CNN), the model achieves high accuracy in classifying these images and provides confidence scores for predictions.

### Key Features
- **Deep Learning Model**: Built with TensorFlow and Keras for robust image classification.
- **Dataset**: Utilizes the PlantVillage dataset with 2,152 images across three classes.
- **Visualization**: Displays sample predictions with actual and predicted labels.
- **Model Saving**: Automatically saves trained models with version control.

---

## 📋 Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

---

## 🛠 Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/potato-disease-classification.git
   cd potato-disease-classification
   ```

2. **Install Dependencies**:
   Ensure Python 3.11+ is installed, then install the required packages:
   ```bash
   pip install tensorflow matplotlib numpy
   ```

3. **Download the Dataset**:
   - Download the **PlantVillage** dataset (Potato subset) from [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) or the [PlantVillage website](https://plantvillage.psu.edu/).
   - Place the dataset in the `PlantVillage` directory.

4. **Jupyter Notebook**:
   Install Jupyter Notebook to run the `model.ipynb` file:
   ```bash
   pip install jupyter
   jupyter notebook
   ```

---

## 📊 Dataset

The dataset is sourced from the **PlantVillage** dataset, available on [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) or the [PlantVillage website](https://plantvillage.psu.edu/). It contains 2,152 images of potato leaves across three classes:
- **Potato___Early_blight**
- **Potato___Late_blight**
- **Potato___healthy**

### Dataset Configuration
- **Image Size**: 256x256 pixels
- **Batch Size**: 32
- **Color Channels**: 3 (RGB)
- **Training Epochs**: 10

The dataset is loaded using TensorFlow's `image_dataset_from_directory` function with shuffling enabled for randomization.

---

## 📈 Usage

To train and evaluate the model:

1. **Open the Notebook**:
   Launch `model.ipynb` in Jupyter Notebook.

2. **Run the Cells**:
   - Import libraries and define hyperparameters.
   - Load and preprocess the dataset.
   - Train the CNN model.
   - Visualize predictions on a sample batch.
   - Save the trained model to the `models` directory.

3. **Example Output**:
   The notebook displays a 3x3 grid of test images, showing:
   - Actual class (e.g., *Potato___Early_blight*)
   - Predicted class
   - Confidence score (in percentage)

4. **Model Saving**:
   Models are saved as `models/{version}.keras`, with versioning for easy tracking.

---

## 🧠 Model Details

The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. Key hyperparameters:
- **Image Size**: 256x256 pixels
- **Batch Size**: 32
- **Channels**: 3 (RGB)
- **Epochs**: 10

The model architecture is defined in `model.ipynb` and trained on the PlantVillage dataset. It outputs probabilities for each class, selecting the one with the highest confidence.

### Prediction Function
The `predict` function:
- Takes an input image.
- Returns the predicted class and confidence score.

---

## 📉 Results

The model delivers high accuracy on the test set. The notebook visualizes predictions in a 3x3 grid, comparing actual vs. predicted labels with confidence scores.

![Sample Predictions](sample_predictions.png)

*Note*: Accuracy varies based on training runs and dataset splits. Check the notebook for detailed metrics.

---

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

Ensure your code adheres to the project's coding standards and includes documentation.

---

## 🌟 Acknowledgments
- **PlantVillage Dataset**: For providing the potato leaf images.
- **TensorFlow & Keras**: For the robust deep learning framework.
- **Matplotlib**: For visualization capabilities.

Happy farming and coding! 🌿

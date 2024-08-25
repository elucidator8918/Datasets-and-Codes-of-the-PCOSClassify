# Benchmarking-Codes-of-the-PCOSGen: Binary Classification

This directory contains the benchmarking codes and evaluation scripts for the PCOSGen Binary Classification Dataset. The PCOSGen dataset is utilized to test and benchmark various binary classification models. The provided codes enable the replication of experiments, evaluation of model performance, and comparison of different algorithms on this specific dataset.

## Running Experiments

Certainly. Here's a comprehensive breakdown of the running experiments, detailing each file in this directory:

### Binary Classification

- **Notebook: `ML_Models_1.ipynb`**  
  **Description**: Implements and evaluates multiple classical machine learning models, including Random Forest, Ridge Classifier, Bagging Classifier (ExtraTreeClassifier), MLPClassifier, KNeighborsClassifier, DecisionTreeClassifier, AdaBoostClassifier, and GaussianNB for binary PCOS classification.

- **Notebook: `ML_Models_2.ipynb`**  
  **Description**: Focuses on advanced gradient boosting models, specifically XGBClassifier and LGBMClassifier, for binary PCOS classification.

- **Notebook: `7_Transfer_Learning_Models.ipynb`**  
  **Description**: Implements and compares seven pre-trained deep learning models (InceptionResNetV2, InceptionV3, MobileNetV2, NasNetMobile, ResNet50V2, VGG19, Xception) for transfer learning in binary PCOS classification.

- **Notebook: `DenseNet169.ipynb`**  
  **Description**: Applies the DenseNet169 architecture for binary PCOS classification using transfer learning.

- **Notebook: `ConvNext.ipynb`**  
  **Description**: Implements the ConvNeXt Base model for binary PCOS classification using transfer learning techniques.

#### run.sh:
Description: A shell script for the creation of setup required for experiments.

## Installing the Dataset

You have two options for installing the dataset: downloading directly or using the Zenodo API.

### 1. Direct Download

You can download the datasets directly from the following links:

- **PCOSGen Training Dataset**: [Download here](https://zenodo.org/records/10430727)
- **PCOSGen Testing Dataset**: [Download here](https://zenodo.org/records/10960327)

After downloading, make sure to unzip the files:

```bash
unzip PCOSGen-train.zip
unzip PCOSGen-test.zip
```

### 2. Using Zenodo API

If you prefer to use the Zenodo API, run the following commands:

```bash
zenodo_get https://doi.org/10.5281/zenodo.10430727
zenodo_get https://doi.org/10.5281/zenodo.10960327
unzip PCOSGen-train.zip
unzip PCOSGen-test.zip
rm PCOSGen-train.zip PCOSGen-test.zip
```

This will download and unzip the dataset files for you.
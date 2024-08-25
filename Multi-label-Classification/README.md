# Benchmarking-Codes-of-the-PCOSGen: Multi-Label Classification

This directory contains the benchmarking codes and evaluation scripts for the PCOSGen Multi-Label Classification Dataset. The PCOSGen dataset is utilized to test and benchmark various Multi-Label classification models. The provided codes enable the replication of experiments, evaluation of model performance, and comparison of different algorithms on this specific dataset.

## Running Experiments

Certainly. Here's a comprehensive breakdown of the running experiments, detailing each file in this directory:

### Multi-Label Classification

- **Notebook: `Multilabel-Classification/multilabel_ML.ipynb`**  
  **Description**: Implements ten machine learning models, including Random Forest, Ridge, Bagging, MLP, KNN, Decision Tree, AdaBoost, Gaussian NB, XGBoost, and LightGBM, for multi-label PCOS classification.

- **Notebook: `Multilabel-Classification/multilabel_TL_ConvNeXtBase.ipynb`**  
  **Description**: Utilizes the ConvNeXtBase architecture for multi-label PCOS classification using transfer learning.

- **Notebook: `Multilabel-Classification/multilabel_TL_DenseNet169.ipynb`**  
  **Description**: Applies the DenseNet169 architecture for multi-label PCOS classification using transfer learning.

- **Notebook: `Multilabel-Classification/multilabel_TL_EfficientNetB7.ipynb`**  
  **Description**: Implements the EfficientNetB7 architecture for multi-label PCOS classification through transfer learning.

- **Notebook: `Multilabel-Classification/multilabel_TL_InceptionResNetV2.ipynb`**  
  **Description**: Leverages the InceptionResNetV2 architecture for multi-label PCOS classification with transfer learning.

- **Notebook: `Multilabel-Classification/multilabel_TL_InceptionV3.ipynb`**  
  **Description**: Uses the InceptionV3 architecture for multi-label PCOS classification via transfer learning.

- **Notebook: `Multilabel-Classification/multilabel_TL_MobileNetV2.ipynb`**  
  **Description**: Employs the MobileNetV2 architecture for multi-label PCOS classification using transfer learning.

- **Notebook: `Multilabel-Classification/multilabel_TL_NASNetMobile.ipynb`**  
  **Description**: Applies the NASNetMobile architecture for multi-label PCOS classification through transfer learning.

- **Notebook: `Multilabel-Classification/multilabel_TL_ResNet50V2.ipynb`**  
  **Description**: Implements the ResNet50V2 architecture for multi-label PCOS classification with transfer learning.

- **Notebook: `Multilabel-Classification/multilabel_TL_VGG19.ipynb`**  
  **Description**: Uses the VGG19 architecture for multi-label PCOS classification through transfer learning.

- **Notebook: `Multilabel-Classification/multilabel_TL_Xception.ipynb`**  
  **Description**: Utilizes the Xception architecture for multi-label PCOS classification using transfer learning.

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
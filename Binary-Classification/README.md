# Benchmarking-Codes-of-the-PCOSGen: Binary Classification

This directory contains the benchmarking codes and evaluation scripts for the PCOSGen Binary Classification Dataset. The PCOSGen dataset is utilized to test and benchmark various binary classification models. The provided codes enable the replication of experiments, evaluation of model performance, and comparison of different algorithms on this specific dataset.

## Running Experiments

Certainly. Here's a comprehensive breakdown of the running experiments, detailing each file in this directory:

### Binary Classification

- **Notebook: `ML_Models.ipynb`**  
  **Description**: Implements and evaluates ten classical machine learning models, including Random Forest, Ridge Classifier, Bagging Classifier (ExtraTreeClassifier), MLPClassifier, KNeighborsClassifier, DecisionTreeClassifier, AdaBoostClassifier, GaussianNB, XGBClassifier and LGBMClassifier for binary PCOS classification.

- **Notebook: `TL_Models.ipynb`**  
  **Description**: Implements and compares nine pre-trained deep learning models (ConvNeXtBase, DenseNet169, InceptionResNetV2, InceptionV3, MobileNetV2, NasNetMobile, ResNet50V2, VGG19, Xception) for transfer learning in binary PCOS classification.

#### run.sh:
Description: A shell script for the creation of setup required for experiments.

## Usage via Terminal

To run the machine learning models for binary classification:

```bash
python Binary-Classification/ML_Models.py --train_excel path/to/train_data.xlsx --test_csv path/to/test_data.csv --train_image_dir path/to/train_images --test_image_dir path/to/test_images
```

Example:
```bash
python Binary-Classification/ML_Models.py --train_excel dataset/train/class_label.xlsx --test_csv dataset/test_label_binary.csv --train_image_dir dataset/train/images --test_image_dir dataset/test/images
```

#### Transfer Learning Models

To run the transfer learning models for binary classification:

```bash
python Binary-Classification/TL_Models.py --train_path path/to/train_data.xlsx --test_path path/to/test_data.csv --image_dir path/to/images --model_name MODEL_NAME
```

Example:
```bash
python Binary-Classification/TL_Models.py --train_path dataset/train/class_label.xlsx --test_path dataset/test_label_binary.csv --image_dir dataset/train/images --model_name ConvNeXtBase
```

Replace `MODEL_NAME` with one of: ConvNeXtBase, DenseNet169, InceptionResNetV2, InceptionV3, MobileNetV2, NASNetMobile, ResNet50V2, VGG19, Xception.

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
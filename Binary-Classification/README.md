# Binary Classification

This directory contains the benchmarking codes and evaluation scripts for the PCOSClassify Binary Classification Dataset. The PCOSClassify dataset is utilized to test and benchmark various binary classification models. The provided codes enable the replication of experiments, evaluation of model performance, and comparison of different algorithms on this specific dataset.

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


### Installing the Dataset

#### 1. Direct Download

You can download the datasets directly from the following links:

- **PCOSClassify Dataset**: [Download here](https://figshare.com/ndownloader/files/50173173)

After downloading, make sure to unzip the files:

```bash
unzip PCOSClassify\ Data.zip
```

This will download and unzip the dataset files for you.

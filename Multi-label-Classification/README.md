# Multi-label Classification

This directory contains the benchmarking codes and evaluation scripts for the PCOSClassify Multi-Label Classification Dataset. The PCOSClassify dataset is utilized to test and benchmark various Multi-Label classification models. The provided codes enable the replication of experiments, evaluation of model performance, and comparison of different algorithms on this specific dataset.

## Running Experiments

Certainly. Here's a comprehensive breakdown of the running experiments, detailing each file in this directory:

### Multi-Label Classification

- **Notebook: `Multilabel-Classification/ML_Models.ipynb`**  
  **Description**: Implements ten machine learning models, including Random Forest, Ridge, Bagging, MLP, KNN, Decision Tree, AdaBoost, Gaussian NB, XGBoost, and LightGBM, for multi-label PCOS classification.

- **Notebook: `Multilabel-Classification/TL_Models.ipynb`**  
  **Description**: Utilizes nine pretrained models including ConvNeXtBase, DenseNet169, InceptionResNetV2, InceptionV3, MobileNetV2, NASNetMobile, ResNet50V2, VGG19 and Xception architectures for multi-label PCOS classification using transfer learning.

#### run.sh:
Description: A shell script for the creation of setup required for experiments.

## Usage via Terminal

To run the machine learning models for multi-label classification:

```bash
python Multi-label-Classification/ML_Models.py --train_val_path path/to/train_val_data.xlsx --test_path path/to/test_data.csv --image_dir path/to/images --image_size IMAGE_SIZE
```

Example:
```bash
python Multi-label-Classification/ML_Models.py --train_val_path dataset/train_val/multilabelpcos.xlsx --test_path dataset/test/test_label_multi.csv --image_dir dataset/train_val/images --image_size 300
```

#### Transfer Learning Models

To run the transfer learning models for multi-label classification:

```bash
python Multi-label-Classification/TL_Models.py --excel_path path/to/train_val_data.xlsx --csv_path path/to/test_data.csv --image_dir path/to/train_images --test_image_dir path/to/test_images --model MODEL_NAME --batch_size BATCH_SIZE --epochs EPOCHS
```

Example:
```bash
python Multi-label-Classification/TL_Models.py --excel_path dataset/train_val/multilabelpcos.xlsx --csv_path dataset/test/test_label_multi.csv --image_dir dataset/train_val/images --test_image_dir dataset/test/images --model ConvNeXtBase --batch_size 32 --epochs 250
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

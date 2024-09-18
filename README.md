# Benchmarking-Codes-of-the-PCOSGen: Binary & Multilabel Classification

![Workflow Architecture](Workflow_Architecture.png)

## Introduction

This repository contains comprehensive benchmarking codes for automated Polycystic Ovary Syndrome (PCOS) detection using ultrasound images. Our study investigates the performance of various machine learning algorithms and transfer learning approaches on the novel PCOSGen dataset.

The Auto-PCOS classification challenge aims to foster the development, testing, and evaluation of Artificial Intelligence (AI) models for automatic PCOS classification. Using healthy and unhealthy frames extracted from ultrasound videos, this challenge encourages the creation of vendor-agnostic, interpretable, and broadly applicable AI models.

The PCOSGen dataset, a first-of-its-kind resource, comprises diverse training and test datasets collected from multiple internet sources, including YouTube, ultrasoundcases.info, and Kaggle. It has been meticulously annotated by experienced gynecologists based in New Delhi, India, specifically for this research.

Our study evaluates the following models:

### Machine Learning Algorithms

1. Random Forest Classifier
2. Ridge Classifier
3. Bagging Classifier (with Extra Tree Classifier)
4. Multi-layer Perceptron Classifier
5. K-Nearest Neighbors Classifier
6. Decision Tree Classifier
7. AdaBoost Classifier
8. Gaussian Naive Bayes
9. XGBoost Classifier
10. LightGBM Classifier

#### Binary Classification: 

- All models can be found here - `/Binary-Classification/ML_Models.ipynb`

#### Multi-label Classification: 

- All models can be found here - `/Multi-label-Classification/ML_Models.ipynb`

### Transfer Learning (Deep Learning Architectures)

1. InceptionResNetV2
2. InceptionV3
3. MobileNetV2
4. NASNetMobile
5. ResNet50V2
6. VGG19
7. Xception
8. DenseNet169
9. ConvNeXtBase

#### Binary Classification:

- All models can be found here - `/Binary-Classification/TL_Models.ipynb`

#### Multi-label Classification:

- All models can be found here - `/Multi-label-Classification/TL_Models.ipynb`

Our findings suggest that both machine learning and transfer learning approaches hold significant promise for automated PCOS detection. However, further research is needed to optimize these techniques and improve their clinical applicability.

A comprehensive technical paper detailing our methodology, results, and insights is currently in preparation and will be available soon - Link - {WIP}

## Repository Structure

```
.
├── Binary-Classification/
│   ├── ML_Models.ipynb
│   ├── ML_Models.py
│   ├── TL_Models.ipynb
│   ├── TL_Models.py
│   └── README.md
├── Multi-label-Classification/
│   ├── ML_Models.ipynb
│   ├── ML_Models.py
│   ├── TL_Models.ipynb
│   ├── TL_Models.py
│   └── README.md
├── Tabular Dataset/
├── LICENSE
├── README.md
├── run.sh
└── Workflow_Architecture.png
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/elucidator8918/Benchmarking-Codes-of-the-PCOSGen.git
cd Benchmarking-Codes-of-the-PCOSGen
```

### Install Dependencies

```bash
conda create -n PCOSGen python=3.10.12 anaconda
conda init
conda activate PCOSGen
bash run.sh
```

### Installing the Dataset

You have two options for installing the dataset: downloading directly or using the Zenodo API.

#### 1. Direct Download

You can download the datasets directly from the following links:

- **PCOSGen Training Dataset**: [Download here](https://zenodo.org/records/10430727)
- **PCOSGen Testing Dataset**: [Download here](https://zenodo.org/records/10960327)

After downloading, make sure to unzip the files:

```bash
unzip PCOSGen-train.zip
unzip PCOSGen-test.zip
```

#### 2. Using Zenodo API

If you prefer to use the Zenodo API, run the following commands:

```bash
zenodo_get https://doi.org/10.5281/zenodo.10430727
zenodo_get https://doi.org/10.5281/zenodo.10960327
unzip PCOSGen-train.zip
unzip PCOSGen-test.zip
rm PCOSGen-train.zip PCOSGen-test.zip
```

This will download and unzip the dataset files for you.

## Running Experiments

Certainly. Here's a comprehensive breakdown of the running experiments, detailing each file in the repository:

### Binary Classification

- **Notebook: `ML_Models.ipynb`**  
  **Description**: Implements and evaluates ten classical machine learning models, including Random Forest, Ridge Classifier, Bagging Classifier (ExtraTreeClassifier), MLPClassifier, KNeighborsClassifier, DecisionTreeClassifier, AdaBoostClassifier, GaussianNB, XGBClassifier and LGBMClassifier for binary PCOS classification.

- **Notebook: `TL_Models.ipynb`**  
  **Description**: Implements and compares nine pre-trained deep learning models (ConvNeXtBase, DenseNet169, InceptionResNetV2, InceptionV3, MobileNetV2, NasNetMobile, ResNet50V2, VGG19, Xception) for transfer learning in binary PCOS classification.

### Multi-label Classification

- **Notebook: `Multilabel-Classification/ML_Models.ipynb`**  
  **Description**: Implements ten machine learning models, including Random Forest, Ridge, Bagging, MLP, KNN, Decision Tree, AdaBoost, Gaussian NB, XGBoost, and LightGBM, for multi-label PCOS classification.

- **Notebook: `Multilabel-Classification/TL_Models.ipynb`**  
  **Description**: Utilizes nine pretrained models including ConvNeXtBase, DenseNet169, InceptionResNetV2, InceptionV3, MobileNetV2, NASNetMobile, ResNet50V2, VGG19 and Xception architectures for multi-label PCOS classification using transfer learning.

#### run.sh:
Description: A shell script for the creation of setup required for experiments.

### Tabular Dataset


## Usage via Terminal

### Binary Classification

#### Machine Learning Models

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

### Multi-label Classification

#### Machine Learning Models

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

## Results

The preview of the comprehensive results section of our technical paper can be seen here as follows:

### Training Data

| Metric                                 | VGG19 | Xception | ResNet50V2 | InceptionV3 | InceptionResNetV2 | MobileNetV2 | DenseNet169 | NASNetMobile | ConvNeXtBase |
|----------------------------------------|---------|----------|------------|-------------|-------------------|-------------|-------------|--------------|--------------|
| **Overall Accuracy (avg. of 250 epochs)** | 0.90    | 0.94     | 0.97       | 0.95        | 0.95              | 0.97        | 0.95        | 0.97         | 0.90         |
| **Overall Accuracy (last epoch)**       | 0.96    | 0.98     | 0.99       | 0.99        | 0.99              | 0.98        | 0.98        | 0.99         | 0.94         |
| **Binary Accuracy (avg. of 250 epochs)**| 0.90    | 0.94     | 0.97       | 0.95        | 0.95              | 0.97        | 0.95        | 0.97         | 0.90         |
| **Binary Accuracy (last epoch)**        | 0.96    | 0.98     | 0.99       | 0.99        | 0.99              | 0.98        | 0.98        | 0.99         | 0.94         |
| **Overall Categorical Accuracy (avg.)** | 1.00    | 1.00     | 1.00       | 1.00        | 1.00              | 1.00        | 1.00        | 1.00         | 1.00         |
| **Overall Categorical Accuracy (last)** | 1.00    | 1.00     | 1.00       | 1.00        | 1.00              | 1.00        | 1.00        | 1.00         | 1.00         |
| **Top-k Categorical Accuracy (avg.)**   | 1.00    | 1.00     | 1.00       | 1.00        | 1.00              | 1.00        | 1.00        | 1.00         | 1.00         |
| **Top-k Categorical Accuracy (last)**   | 1.00    | 1.00     | 1.00       | 1.00        | 1.00              | 1.00        | 1.00        | 1.00         | 1.00         |
| **Precision (avg. of 250 epochs)**      | 0.84    | 0.90     | 0.95       | 0.92        | 0.92              | 0.94        | 0.90        | 0.94         | 0.83         |
| **Precision (last epoch)**              | 0.95    | 0.96     | 0.98       | 0.98        | 0.98              | 0.97        | 0.96        | 0.97         | 0.90         |
| **Recall (avg. of 250 epochs)**         | 0.81    | 0.90     | 0.95       | 0.92        | 0.92              | 0.94        | 0.90        | 0.94         | 0.83         |
| **Recall (last epoch)**                | 0.92    | 0.96     | 0.98       | 0.98        | 0.98              | 0.97        | 0.97        | 0.97         | 0.89         |
| **Loss (avg. of 250 epochs)**           | 0.24    | 0.86     | 1.19       | 1.12        | 0.69              | 0.93        | 0.86        | 0.38         | 0.69         |
| **Loss (last epoch)**                  | 0.10    | 0.57     | 1.35       | 0.67        | 0.37              | 0.87        | 0.52        | 0.39         | 0.49         |

### Validation Data

| Metric                                 | VGG19 | Xception | ResNet50V2 | InceptionV3 | InceptionResNetV2 | MobileNetV2 | DenseNet169 | NASNetMobile | ConvNeXtBase |
|----------------------------------------|---------|----------|------------|-------------|-------------------|-------------|-------------|--------------|--------------|
| **Overall Accuracy (avg. of 250 epochs)** | 0.72    | 0.72     | 0.77       | 0.74        | 0.72              | 0.71        | 0.73        | 0.72         | 0.68         |
| **Overall Accuracy (last epoch)**       | 0.75    | 0.75     | 0.79       | 0.72        | 0.68              | 0.70        | 0.77        | 0.67         | 0.71         |
| **Binary Accuracy (avg. of 250 epochs)**| 0.72    | 0.72     | 0.77       | 0.74        | 0.72              | 0.71        | 0.73        | 0.72         | 0.68         |
| **Binary Accuracy (last epoch)**        | 0.75    | 0.75     | 0.79       | 0.72        | 0.68              | 0.70        | 0.77        | 0.67         | 0.71         |
| **Overall Categorical Accuracy (avg.)** | 1.00    | 1.00     | 1.00       | 1.00        | 1.00              | 1.00        | 1.00        | 1.00         | 1.00         |
| **Overall Categorical Accuracy (last)** | 1.00    | 1.00     | 1.00       | 1.00        | 1.00              | 1.00        | 1.00        | 1.00         | 1.00         |
| **Top-k Categorical Accuracy (avg.)**   | 1.00    | 1.00     | 1.00       | 1.00        | 1.00              | 1.00        | 1.00        | 1.00         | 1.00         |
| **Top-k Categorical Accuracy (last)**   | 1.00    | 1.00     | 1.00       | 1.00        | 1.00              | 1.00        | 1.00        | 1.00         | 1.00         |
| **Precision (avg. of 250 epochs)**      | 0.57    | 0.57     | 0.60       | 0.56        | 0.54              | 0.48        | 0.53        | 0.55         | 0.48         |
| **Precision (last epoch)**              | 0.61    | 0.63     | 0.64       | 0.50        | 0.46              | 0.45        | 0.64        | 0.48         | 0.50         |
| **Recall (avg. of 250 epochs)**         | 0.57    | 0.53     | 0.39       | 0.42        | 0.48              | 0.55        | 0.50        | 0.60         | 0.51         |
| **Recall (last epoch)**                | 0.45    | 0.48     | 0.38       | 0.57        | 0.63              | 0.45        | 0.35        | 0.72         | 0.39         |
| **Loss (avg. of 250 epochs)**           | 1.20    | 12.87    | 27.21      | 21.32       | 12.47             | 19.09       | 13.01       | 7.20         | 5.72         |
| **Loss (last epoch)**                  | 1.55    | 17.80    | 35.89      | 25.14       | 18.29             | 24.75       | 19.52       | 10.13        | 7.38         |

## Setup used for Evaluation

The models were trained for 250 epochs with minimal preprocessing, using a 40 GB DGX A100 NVIDIA GPU workstation provided by the Department of Electronics and Communication Engineering at Indira Gandhi Delhi Technical University for Women, New Delhi.

## Contributions

Palak Handa conceptualized the research idea, was involved in data collection, and manuscript writing. Anushka Saini performed the benchmarking, contributed in writing the initial draft of the manuscript. Siddhant Dutta was involved in data wrangling, analysis, and contributed in initial manuscript writing and github development. Nidhi Choudhary was involved in suggestions and performed the medical annotations. Ammireza Mahbod, Florian Schwarzhans, and Ramona Woitek were involved in suggestions and manuscript reviewing. Nidhi Goel was involved in the project administration. The authors are thankful to Harsh Pathak for his initial attempt for data collection, Charvi Bansal for helping in initial stages of the benchmarking and manuscript writing, Nikita Garg for developing the FEM-AI Labeller application and Manya Joshi for putting it on github. The PCOSGen-train and PCOS-Gen-test datasets have been actively downloaded more than 1000 times and were utilized in the Auto-PCOS Classification challenge. The challenge page is available here - [Visit the Link](https://misahub.in/pcos/index.html)

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
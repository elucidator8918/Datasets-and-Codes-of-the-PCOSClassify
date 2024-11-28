# Datasets and Codes of the PCOSClassify: An Ultrasound Imaging Dataset and Benchmark for Machine Learning Classification of PCOS

This repository contains the datasets and codes utilized in the manuscript _**'PCOSClassify: An Ultrasound Imaging Dataset and Benchmark for Machine Learning Classification of PCOS'**_. 

PCOSClassify consists of two different types of datasets. The first dataset consists of ultrasound images of healthy and un-healthy females in the context of PCOS disease with binary and multi-labels. The binary labels and its ultrasound images were utilized as a part of the Auto-PCOS Classification Challenge. The multi-labels includes Round and Thin walled with posterior enhancement, Cumulus oophorous, Corpus luteum, Hemorrhagic ovarian cyst, Hemorrhagic corpus luteum, Endometrioma, Serous cystadenoma, Serous cystadenocarcinoma, Mucinous cystadenoma, Mucinous cystadenocarcinoma, Dermoid cyst, Dermoid plug, Rokitansky nodule, Dermoid mesh, Dot dash pattern, Floating balls sign, Ovarian fibroma, Ovarian thecoma, Metastasis, Para ovarian cyst, Polycystic ovary, Ovarian hyperstimulation syndrome, Ovarian torsion, Thick hyperechoic margin, Vaginal ultrasound, Right ovary, Transvaginal ultrasound, Gestational sac, Foetus, Chocolate cyst, Cervix, Urinary bladder, Polyp, Cervical cyst, Adnexa, Vagina, Ovary, and Uterus. The second dataset consists of a tabular data which was collected through a research survey of 242 women aged 18 to 45 years, conducted to investigate menstrual cycles and hygiene among women from various geographical regions across India, focusing on different age groups. It consists of questions and answers about menstrual cycles, lifestyle factors, and hygiene practices of healthy and un-healthy females in the context of PCOS disease. Both the datasets are available (here)[https://figshare.com/articles/dataset/Datasets_of_the_PCOSClassify/27600816?file=50173173]













# Datasets and Codes of the PCOSClassify: An Ultrasound Imaging Dataset and Benchmark for Machine Learning Classification of PCOS

This repository contains the datasets and codes utilized in the manuscript _**'PCOSClassify: An Ultrasound Imaging Dataset and Benchmark for Machine Learning Classification of PCOS'**_. 


https://figshare.com/articles/dataset/Datasets_of_the_PCOSClassify/27600816?file=50173173
At first, the codes used to benchmark the PCOSClassify train and test dataset have been discussed. The PCOSGen train and test dataset contains ultrasound images with binary labels. This dataset was also utilized in the Auto-PCOS Classification challenge. An extended version of the PCOSGen train and test dataset containing multi-labels has also been released and benchmarked. The benchmarking codes of the multi-label classification have been put in a second folder. A third dataset, the tabular dataset collected by the team has also been released and discussed in this GitHub repository. 

## Introduction

The PCOSGen dataset, a first-of-its-kind resource, comprises diverse training and test datasets collected from multiple internet sources, including YouTube, ultrasoundcases.info, and Kaggle. It has been meticulously annotated by experienced gynaecologists based in New Delhi, India, specifically for this research.

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

#### Tabular Dataset:

- A Detailed Description regarding the research conducted could be found here - `/Tabular-Dataset/README.md`

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

Note: It only contains the binary labels PCOS and Non-PCOS.
The full PCOS-Gen dataset which contains binary and multi-labels is available [here](https://doi.org/10.6084/m9.figshare.27106024.v1). 
Similar steps can be followed to run the multi-label-based dataset. 

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
  **Description**: Utilizes nine pre-trained models including ConvNeXtBase, DenseNet169, InceptionResNetV2, InceptionV3, MobileNetV2, NASNetMobile, ResNet50V2, VGG19 and Xception architectures for multi-label PCOS classification using transfer learning.

#### run.sh:
Description: A shell script for the creation of setup required for experiments.

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

## Setup used for Evaluation

All models were trained for 250 epochs with minimal preprocessing, using a 40 GB DGX A100 NVIDIA GPU workstation provided by the Department of Electronics and Communication Engineering at Indira Gandhi Delhi Technical University for Women, New Delhi.

### Developed Tabular Dataset for PCOS 

## Contributions

Palak Handa conceptualized the research idea, was involved in data collection, and manuscript writing. Anushka Saini performed the benchmarking and contributed in writing the initial draft of the manuscript. Siddhant Dutta was involved in data wrangling, analysis, and contributed to initial manuscript writing and GitHub development. Nishi Choudhary was involved in suggestions and performed the medical annotations of the PCOSGen train and test dataset. Ammireza Mahbod, Florian Schwarzhans, and Ramona Woitek were involved in suggestions and manuscript reviewing. Nidhi Goel was involved in the project administration. The authors are thankful to Harsh Pathak for his initial attempt for data collection, Charvi Bansal for helping in the initial stages of the benchmarking and manuscript writing, Nikita Garg for developing the FEM-AI Labeller application and Manya Joshi for putting it on GitHub. The tabular dataset was collected by MISAHUB Research Team: Mahima Singh, Bandana Pal , Naila Jan, Anaa Makhdoomi, Esha Gupta, and Muskan Gupta.The PCOSGen-train and PCOS-Gen-test datasets have been actively downloaded more than 1000 times and were utilized in the Auto-PCOS Classification challenge. The challenge page is available here - [Visit the Link](https://misahub.in/pcos/index.html)

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

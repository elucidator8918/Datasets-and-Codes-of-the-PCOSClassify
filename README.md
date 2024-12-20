# Datasets and Codes of the PCOSClassify: An Ultrasound Imaging Dataset and Benchmark for Machine Learning Classification of PCOS

This repository provides the datasets and codes used in the manuscript titled "PCOSClassify: An Ultrasound Imaging Dataset and Benchmark for Machine Learning Classification of PCOS."

PCOSClassify includes two types of datasets:

## Ultrasound Imaging Dataset:

This dataset contains ultrasound images categorized as healthy and unhealthy in the context of PCOS (Polycystic Ovary Syndrome). It includes binary and multi-label classifications:
### Binary Labels: 
- These datasets, named PCOSGen-train and PCOSGen-test, have been actively downloaded over 1,000 times and were utilized in the Auto-PCOS Classification Challenge. The challenge page is available [here](https://misahub.in/pcos/index.html).

### Multi-label Annotations: 
- Each ultrasound image includes multiple labels such as Round and Thin Walled with Posterior Enhancement, Cumulus Oophorous, Corpus Luteum, Hemorrhagic Ovarian Cyst, Endometrioma, Serous Cystadenoma, Ovarian Fibroma, Ovarian Hyperstimulation Syndrome, Gestational Sac, Chocolate Cyst, Polyp, Vaginal Ultrasound, and many more.

Both the binary and multi-label datasets are derived from the same ultrasound images. This dataset is the first of its kind, comprising diverse training and test sets collected from various internet sources, including YouTube, ultrasoundcases.info, and Kaggle. It has been meticulously annotated by an experienced gynecologist based in New Delhi, India, specifically for this research.

## Tabular Dataset:

This dataset was collected through a research survey of 242 women aged 18 to 45 years. The survey focused on investigating menstrual cycles, lifestyle factors, and hygiene practices across different age groups and geographical regions of India. It includes comprehensive data on menstrual health and PCOS-related conditions, providing a valuable resource for understanding the correlation between lifestyle and PCOS.
Both the datasets are available [here](https://figshare.com/articles/dataset/Datasets_of_the_PCOSClassify/27600816?file=50173173).

The ultrasound imaging dataset has been used for automatic binary and multi-label classification using standard AI models, such as:

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

- The models can be found here - `/Binary-Classification/ML_Models.ipynb`

#### Multi-label Classification: 

- The models can be found here - `/Multi-label-Classification/ML_Models.ipynb`

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

- The models can be found here - `/Binary-Classification/TL_Models.ipynb`

#### Multi-label Classification:

- The models can be found here - `/Multi-label-Classification/TL_Models.ipynb`

#### Tabular Dataset:

- A Detailed Description regarding the research conducted could be found here - `/Tabular-Dataset/README.md`

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

#### 1. Direct Download

You can download the datasets directly from the following links:

- **PCOSClassify Dataset**: [Download here](https://figshare.com/ndownloader/files/50173173)

After downloading, make sure to unzip the files:

```bash
unzip PCOSClassify\ Data.zip
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

All models were trained for 250 epochs with minimal preprocessing, using a 40 GB DGX A100 NVIDIA GPU workstation provided by the Department of Electronics and Communication Engineering at Indira Gandhi Delhi Technical University for Women, New Delhi, India.

## Contributions

Palak Handa conceptualized the research idea, was involved in data collection, and manuscript writing. Anushka Saini performed the benchmarking and contributed in writing the initial draft of the manuscript. Siddhant Dutta was involved in data wrangling, analysis, and contributed to initial manuscript writing and GitHub development. Nishi Choudhary was involved in manuscript reviewing, suggestions and extensively performed the medical annotations of the first dataset. Ramona Woitek was involved in suggestions and manuscript reviewing. Nidhi Goel was involved in the project administration. The authors are thankful to Harsh Pathak for his initial attempt for data collection, Charvi Bansal for helping in the initial stages of the benchmarking and manuscript writing, Nikita Garg for developing the FEM-AI Labeller application and Manya Joshi for putting it on GitHub. The tabular dataset was collected by MISAHUB Research Team: Mahima Singh, Bandana Pal , Naila Jan, Anaa Makhdoomi, Esha Gupta, and Muskan Gupta. 

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details. All the datasets come under CC BY 4.0 NC where they maybe used for research but strictly not utilized for commercial purposes. 

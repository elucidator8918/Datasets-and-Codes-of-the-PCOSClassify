import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, jaccard_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_data(train_val_path, test_path, image_dir, target_size):
    # Load data
    train_val_df = pd.read_excel(train_val_path)
    test_df = pd.read_csv(test_path)
    
    # Clean data
    train_val_df = train_val_df.dropna(how='all').dropna(how='all', axis=1)
    
    # Split train and validation
    train, validate = train_test_split(train_val_df, test_size=0.2, random_state=42)
    
    # Prepare labels
    label_columns = ["Round and Thin", "Cumulus oophorous", "Corpus luteum", "Hemorrhagic ovarian cyst", 
                     "Hemorrhagic corpus luteum", "Endometrioma", "serous cystadenoma", "Serous cystadenocarcinoma", 
                     "Mucinous cystadenoma", "Mucinous cystadenocarcinoma", "Dermoid cyst", "Dermoid plug", 
                     "Rokitansky nodule", "Dermoid mesh", "Dot dash pattern", "Floating balls sign", "Ovarian fibroma", 
                     "Ovarian thecoma", "Metastasis", "Para ovarian cyst", "Polycystic ovary", 
                     "Ovarian hyperstimulation syndrome", "Ovarian torsion", "Thick hyperechoic margin", 
                     "Vaginal ultrasound", "Transvaginal ultrasound", "Gestational sac", "Foetus", "Chocolate cyst", 
                     "Cervix", "Urinary bladder", "Polyp", "Cervical cyst"]
    
    train_labels = train[label_columns].values
    validate_labels = validate[label_columns].values
    test_labels = test_df[label_columns].values
    
    # Load and preprocess images
    def load_images(df, image_dir):
        images = []
        for filename in df['ImagePath']:
            image_path = os.path.join(image_dir, filename)
            image = load_img(image_path, target_size=target_size)
            image = img_to_array(image) / 255.0
            images.append(image)
        return np.array(images, dtype=np.float32)
    
    train_images = load_images(train, image_dir)
    validate_images = load_images(validate, image_dir)
    test_images = load_images(test_df, image_dir)
    
    return train_images, train_labels, validate_images, validate_labels, test_images, test_labels

def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, model_name):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Weighted F1 score: {f1_score(y_test, y_pred, average='weighted') * 100:.2f}%")
    print(f"Weighted recall: {recall_score(y_test, y_pred, average='weighted') * 100:.2f}%")
    print(f"Weighted precision: {precision_score(y_test, y_pred, average='weighted') * 100:.2f}%")
    print(f"Weighted Jaccard score: {jaccard_score(y_test, y_pred, average='weighted') * 100:.2f}%")

def main(args):
    # Load and preprocess data
    train_images, train_labels, validate_images, validate_labels, test_images, test_labels = load_and_preprocess_data(
        args.train_val_path, args.test_path, args.image_dir, (args.image_size, args.image_size)
    )
    
    # Reshape images for classifiers
    x_train = train_images.reshape(train_images.shape[0], -1)
    x_test_internal = validate_images.reshape(validate_images.shape[0], -1)
    x_test_external = test_images.reshape(test_images.shape[0], -1)
    
    # List of classifiers
    classifiers = [
        (RandomForestClassifier(max_depth=2, random_state=0), "Random Forest"),
        (RidgeClassifier(), "Ridge Classifier"),
        (BaggingClassifier(ExtraTreeClassifier(random_state=0), random_state=0), "Bagging Classifier"),
        (MLPClassifier(random_state=1, max_iter=300), "MLP Classifier"),
        (KNeighborsClassifier(n_neighbors=3), "KNN Classifier"),
        (DecisionTreeClassifier(random_state=0), "Decision Tree"),
        (SVC(kernel='rbf', gamma='auto'), "SVC"),
        (GaussianNB(), "Gaussian Naive Bayes"),
        (LogisticRegression(), "Logistic Regression"),
        (AdaBoostClassifier(), "AdaBoost Classifier")
    ]
    
    # Train and evaluate models
    for clf, name in classifiers:
        print(f"\nTraining and evaluating {name}")
        print("Internal validation results:")
        train_and_evaluate_model(clf, x_train, train_labels, x_test_internal, validate_labels, name)
        print("\nExternal test results:")
        train_and_evaluate_model(clf, x_train, train_labels, x_test_external, test_labels, name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilabel Classification for Medical Images")
    parser.add_argument("--train_val_path", type=str, default="/workspace/anushka saini/train_val/multilabelpcos.xlsx", 
                        help="Path to the train/validation Excel file")
    parser.add_argument("--test_path", type=str, default="/workspace/anushka saini/test/test_label_multi.csv", 
                        help="Path to the test CSV file")
    parser.add_argument("--image_dir", type=str, default="/workspace/anushka saini/train_val/images", 
                        help="Directory containing the images")
    parser.add_argument("--image_size", type=int, default=300, 
                        help="Size to resize images (both width and height)")
    
    args = parser.parse_args()
    main(args)
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, jaccard_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import lightgbm as lgb
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_data(train_excel_path, test_csv_path, train_image_dir, test_image_dir):
    # Load class labels
    train_df = pd.read_excel(train_excel_path)
    test_df = pd.read_csv(test_csv_path)

    # Clean the train DataFrame
    train_df = train_df.dropna(how='all').dropna(how='all', axis=1)

    # Prepare image paths and labels
    train_image_paths = [os.path.join(train_image_dir, filename) for filename in train_df['imagePath']]
    test_image_paths = [os.path.join(test_image_dir, filename) for filename in test_df['imagePath']]

    train_labels = train_df[["Healthy"]].values
    test_labels = test_df[["Healthy"]].values

    return train_image_paths, train_labels, test_image_paths, test_labels

def preprocess_images(image_paths, target_size):
    images = []
    for image_path in image_paths:
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image) / 255.0
        images.append(image)
    return np.array(images, dtype=np.float32)

def train_evaluate_model(model, x_train, y_train, x_test, y_test, model_name):
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
    # Global variables and parameters
    params = {
        'image_size': (300, 300),
        'test_size': 0.2,
        'random_state': 0
    }

    # Models configuration
    models = [
        ("Random Forest", RandomForestClassifier(max_depth=2, random_state=params['random_state'])),
        ("Ridge Classifier", RidgeClassifier()),
        ("Bagging with Extra Trees", BaggingClassifier(ExtraTreeClassifier(random_state=params['random_state']), random_state=params['random_state'])),
        ("MLP Classifier", MLPClassifier(random_state=params['random_state'], max_iter=300)),
        ("KNN", KNeighborsClassifier(n_neighbors=3)),
        ("Decision Tree", DecisionTreeClassifier(random_state=params['random_state'])),
        ("AdaBoost", AdaBoostClassifier()),
        ("Gaussian Naive Bayes", GaussianNB()),
        ("XGBoost", XGBClassifier(max_depth=3, learning_rate=0.1, subsample=0.5)),
        ("LightGBM", lgb.LGBMClassifier())
    ]

    # Load and preprocess data
    train_image_paths, train_labels, test_image_paths, test_labels = load_data(
        args.train_excel, args.test_csv, args.train_image_dir, args.test_image_dir
    )
    
    train_images = preprocess_images(train_image_paths, params['image_size'])
    test_images = preprocess_images(test_image_paths, params['image_size'])

    # Split train data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        train_images.reshape(train_images.shape[0], -1),
        train_labels,
        test_size=params['test_size'],
        random_state=params['random_state']
    )

    x_test = test_images.reshape(test_images.shape[0], -1)

    # Train and evaluate models
    for model_name, model in models:
        train_evaluate_model(model, x_train, y_train.ravel(), x_val, y_val.ravel(), f"{model_name} (Validation)")
        train_evaluate_model(model, x_train, y_train.ravel(), x_test, test_labels.ravel(), f"{model_name} (Test)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate multiple ML models on image data")
    parser.add_argument("--train_excel", default="C:/Users/anushka saini/OneDrive/Desktop/AutoPCOS_classification_challenge/dataset/train/class_label.xlsx", 
                        help="Path to the training Excel file")
    parser.add_argument("--test_csv", default="C:/Users/anushka saini/OneDrive/Desktop/AutoPCOS_classification_challenge/dataset/test_label_binary.csv", 
                        help="Path to the test CSV file")
    parser.add_argument("--train_image_dir", default="C:/Users/anushka saini/OneDrive/Desktop/AutoPCOS_classification_challenge/dataset/train/images", 
                        help="Directory containing training images")
    parser.add_argument("--test_image_dir", default="C:/Users/anushka saini/OneDrive/Desktop/AutoPCOS_classification_challenge/dataset/test/images", 
                        help="Directory containing test images")
    
    args = parser.parse_args()
    main(args)
import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import (
    ConvNeXtBase, DenseNet169, InceptionResNetV2, InceptionV3,
    MobileNetV2, NASNetMobile, ResNet50V2, VGG19, Xception
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def load_data(excel_path, csv_path, image_dir):
    la1 = pd.read_excel(excel_path)
    la2 = pd.read_csv(csv_path)
    df = la1.dropna(how='all').dropna(how='all', axis=1)
    return df, la2

def preprocess_images(image_paths, target_size):
    images = []
    for path in image_paths:
        image = load_img(path, target_size=target_size)
        image = img_to_array(image) / 255.0
        images.append(image)
    return np.array(images, dtype=np.float32)

def create_model(base_model_name, input_shape, num_classes):
    base_models = {
        'ConvNeXtBase': ConvNeXtBase,
        'DenseNet169': DenseNet169,
        'InceptionResNetV2': InceptionResNetV2,
        'InceptionV3': InceptionV3,
        'MobileNetV2': MobileNetV2,
        'NASNetMobile': NASNetMobile,
        'ResNet50V2': ResNet50V2,
        'VGG19': VGG19,
        'Xception': Xception
    }
    
    base_model = base_models[base_model_name](weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(), loss='binary_crossentropy', 
                  metrics=['accuracy', 'binary_accuracy', 'categorical_accuracy', 
                           'top_k_categorical_accuracy', 'Precision', 'Recall'])
    return model

def train_model(model, train_images, train_labels, validate_images, validate_labels, 
                batch_size, epochs, model_name):
    csv_logger = CSVLogger(f"{model_name}_model_history_log.csv", append=True)
    
    model.fit(train_images, train_labels,
              validation_data=(validate_images, validate_labels),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[csv_logger],
              shuffle=False)
    
    model.save_weights(f'{model_name}_weights.h5')
    model.save(f'{model_name}_model.h5')

def evaluate_model(model, images, labels):
    labels_pred = model.predict(images)
    labels_pred = np.argmax(labels_pred, axis=1)
    labels = np.argmax(labels, axis=1)
    print(classification_report(labels, labels_pred))

def main(args):
    # Global variables and parameters
    image_size = (300, 300)
    num_classes = 33
    label_columns = ["Round and Thin", "Cumulus oophorous", "Corpus luteum", "Hemorrhagic ovarian cyst", 
                     "Hemorrhagic corpus luteum", "Endometrioma", "serous cystadenoma", "Serous cystadenocarcinoma", 
                     "Mucinous cystadenoma", "Mucinous cystadenocarcinoma", "Dermoid cyst", "Dermoid plug", 
                     "Rokitansky nodule", "Dermoid mesh", "Dot dash pattern", "Floating balls sign", 
                     "Ovarian fibroma", "Ovarian thecoma", "Metastasis", "Para ovarian cyst", "Polycystic ovary", 
                     "Ovarian hyperstimulation syndrome", "Ovarian torsion", "Thick hyperechoic margin", 
                     "Vaginal ultrasound", "Transvaginal ultrasound", "Gestational sac", "Foetus", "Chocolate cyst", 
                     "Cervix", "Urinary bladder", "Polyp", "Cervical cyst"]

    # Load and preprocess data
    df, test_df = load_data(args.excel_path, args.csv_path, args.image_dir)
    train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)

    train_images = preprocess_images([os.path.join(args.image_dir, filename) for filename in train_df['ImagePath']], image_size)
    validate_images = preprocess_images([os.path.join(args.image_dir, filename) for filename in validate_df['ImagePath']], image_size)
    test_images = preprocess_images([os.path.join(args.test_image_dir, filename) for filename in test_df['imagePath']], image_size)

    train_labels = train_df[label_columns].values
    validate_labels = validate_df[label_columns].values
    test_labels = test_df[label_columns].values

    # Create and train model
    model = create_model(args.model, image_size + (3,), num_classes)
    train_model(model, train_images, train_labels, validate_images, validate_labels, 
                args.batch_size, args.epochs, args.model)

    # Evaluate model
    print("Validation Set Evaluation:")
    evaluate_model(model, validate_images, validate_labels)
    print("Test Set Evaluation:")
    evaluate_model(model, test_images, test_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate transfer learning models for image classification")
    parser.add_argument("--excel_path", default="/workspace/anushka saini/train_val/multilabelpcos.xlsx", help="Path to the Excel file containing training data")
    parser.add_argument("--csv_path", default="/workspace/anushka saini/test/test_label_multi.csv", help="Path to the CSV file containing test data")
    parser.add_argument("--image_dir", default="/workspace/anushka saini/train_val/images", help="Directory containing training images")
    parser.add_argument("--test_image_dir", default="/workspace/anushka saini/test/images", help="Directory containing test images")
    parser.add_argument("--model", choices=['ConvNeXtBase', 'DenseNet169', 'InceptionResNetV2', 'InceptionV3', 'MobileNetV2', 'NASNetMobile', 'ResNet50V2', 'VGG19', 'Xception'], default='ConvNeXtBase', help="Transfer learning model to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs for training")

    args = parser.parse_args()
    main(args)
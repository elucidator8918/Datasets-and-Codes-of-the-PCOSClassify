import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50V2, ResNet101V2, InceptionV3, InceptionResNetV2, MobileNetV2, DenseNet169, NASNetMobile, EfficientNetB7, ConvNeXtBase
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

def load_data(train_path, test_path, image_dir):
    # Load and preprocess data
    train_df = pd.read_excel(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df = train_df.dropna(how='all').dropna(how='all', axis=1)
    
    train, validate = train_validate_split(train_df)
    
    train_images, train_labels = preprocess_data(train, image_dir)
    validate_images, validate_labels = preprocess_data(validate, image_dir)
    test_images, test_labels = preprocess_data(test_df, image_dir, is_test=True)
    
    return train_images, train_labels, validate_images, validate_labels, test_images, test_labels

def train_validate_split(df, train_percent=.8, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    return train, validate

def preprocess_data(df, image_dir, is_test=False):
    image_paths = [os.path.join(image_dir, filename) for filename in df['imagePath']]
    images = []
    for image_path in image_paths:
        image = load_img(image_path, target_size=(300, 300))
        image = img_to_array(image) / 255.0
        images.append(image)
    images = np.array(images, dtype=np.float32)
    
    if is_test:
        labels = df[["Healthy"]].values
    else:
        labels = df[["Healthy"]].values
    
    return images, labels

def evaluate_model(model, images, labels, model_name):
    predictions = model.predict(images)
    threshold = 0.5
    predictions_binary = (predictions > threshold).astype(int)
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(labels, predictions_binary))

def main(args):
    # Global variables and parameters
    global_vars = {
        'train_path': args.train_path,
        'test_path': args.test_path,
        'image_dir': args.image_dir,
        'model_paths': {
            'InceptionResNetV2': args.inception_resnet_v2_path,
            'InceptionV3': args.inception_v3_path,
            'MobileNetV2': args.mobile_net_v2_path,
            'NASNetMobile': args.nasnet_mobile_path,
            'ResNet50V2': args.resnet50_v2_path,
            'VGG19': args.vgg19_path,
            'Xception': args.xception_path,
            'ConvNeXtBase': args.convnext_base_path,
            'DenseNet169': args.densenet169_path
        }
    }

    # Load and preprocess data
    train_images, train_labels, validate_images, validate_labels, test_images, test_labels = load_data(
        global_vars['train_path'], global_vars['test_path'], global_vars['image_dir']
    )

    # Evaluate models
    for model_name, model_path in global_vars['model_paths'].items():
        if model_path:
            if model_name in ['ConvNeXtBase', 'DenseNet169']:
                base_model = globals()[model_name](weights='imagenet', include_top=False, input_shape=(300, 300, 3))
                x = base_model.output
                x = Flatten()(x)
                predictions = Dense(1, activation='sigmoid')(x)
                model = Model(inputs=base_model.input, outputs=predictions)
                model.load_weights(model_path)
            else:
                model = load_model(model_path)
            
            print(f"\nEvaluating {model_name} on validation set:")
            evaluate_model(model, validate_images, validate_labels, model_name)
            
            print(f"\nEvaluating {model_name} on test set:")
            evaluate_model(model, test_images, test_labels, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Transfer Learning Models for PCOS Classification")
    parser.add_argument("--train_path", default="/content/drive/MyDrive/PCOS_TL_ML/train/class_label.xlsx", help="Path to training data Excel file")
    parser.add_argument("--test_path", default="/content/drive/MyDrive/PCOS_TL_ML/test/test_label_binary.csv", help="Path to test data CSV file")
    parser.add_argument("--image_dir", default="/content/drive/MyDrive/PCOS_TL_ML/train/images", help="Directory containing images")
    parser.add_argument("--inception_resnet_v2_path", default="/content/drive/MyDrive/PCOS_TL_ML/BinaryLabel/Transfer Learning/InceptionResNetV2/InceptionResNetV2_model.h5")
    parser.add_argument("--inception_v3_path", default="/content/drive/MyDrive/PCOS_TL_ML/BinaryLabel/Transfer Learning/InceptionV3/InceptionV3_model.h5")
    parser.add_argument("--mobile_net_v2_path", default="/content/drive/MyDrive/PCOS_TL_ML/BinaryLabel/Transfer Learning/MobileNetV2/MobileNetV2_model.h5")
    parser.add_argument("--nasnet_mobile_path", default="/content/drive/MyDrive/PCOS_TL_ML/BinaryLabel/Transfer Learning/NasNetMoblie/NASNetMobile_model.h5")
    parser.add_argument("--resnet50_v2_path", default="/content/drive/MyDrive/PCOS_TL_ML/BinaryLabel/Transfer Learning/ResNet50v2/ResNet50V2_model.h5")
    parser.add_argument("--vgg19_path", default="/content/drive/MyDrive/PCOS_TL_ML/BinaryLabel/Transfer Learning/Vgg19/VGG19_model.h5")
    parser.add_argument("--xception_path", default="/content/drive/MyDrive/PCOS_TL_ML/BinaryLabel/Transfer Learning/Xception/Xception_model.h5")
    parser.add_argument("--convnext_base_path", default="/content/drive/MyDrive/PCOS_TL_ML/BinaryLabel/Transfer Learning/ConvNeXtBase/ConvNeXtBase_weights.h5")
    parser.add_argument("--densenet169_path", default="/content/drive/MyDrive/PCOS_TL_ML/BinaryLabel/Transfer Learning/DenseNet169/DenseNet169_weights.h5")
    
    args = parser.parse_args()
    main(args)
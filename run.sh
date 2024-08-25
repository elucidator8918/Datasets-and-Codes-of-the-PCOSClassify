#!/bin/bash

sudo apt-get update && apt-get install libgl1 -y

# Install Python packages
pip install \
    pandas \
    numpy \
    opencv-python \
    pillow \
    matplotlib \
    seaborn \
    scikit-image \
    tensorflow \
    sklearn \
    lime \
    shap \
    openpyxl \
    albumentations \
    sweetviz \
    grad-cam \
    pandas-profiling \
    Keras-Preprocessing \
    zenodo-get
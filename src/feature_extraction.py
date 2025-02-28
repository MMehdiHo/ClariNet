import os
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils import load_and_preprocess_images  # Import from utils

# Constants
IMAGE_SIZE = (512, 512)
IMAGES_PER_FOLDER = 500
CSV_PATH = "../data/balncedata_new_7900.csv"
BASE_PATH = "../data/tiles/"
OUTPUT_PATH = "../output/"

# Load Data
df = pd.read_csv(CSV_PATH)
FOLDERS = {
    "solid": df[df["class"] == "solid"]["image_path"].apply(lambda x: os.path.join(BASE_PATH, x)).tolist(),
    "lepidic": df[df["class"] == "lepidic"]["image_path"].apply(lambda x: os.path.join(BASE_PATH, x)).tolist(),
    "acinar": df[df["class"] == "acinar"]["image_path"].apply(lambda x: os.path.join(BASE_PATH, x)).tolist(),
    "micropapillary": df[df["class"] == "micropapillary"]["image_path"].apply(lambda x: os.path.join(BASE_PATH, x)).tolist(),
    "papillary": df[df["class"] == "papillary"]["image_path"].apply(lambda x: os.path.join(BASE_PATH, x)).tolist(),
}

# Pre-trained Models
MODELS = {
    "vgg16": (VGG16(weights="imagenet", include_top=False, pooling="avg"), preprocess_vgg16),
    "resnet50": (ResNet50(weights="imagenet", include_top=False, pooling="avg"), preprocess_resnet50),
    "inception": (InceptionV3(weights="imagenet", include_top=False, pooling="avg"), preprocess_inception),
}

# Feature Extraction
for model_name, (model, preprocess_func) in MODELS.items():
    print(f"Extracting features with {model_name}...")
    features = []
    labels = []
    for label, (class_name, image_paths) in enumerate(FOLDERS.items()):
        images = load_and_preprocess_images(image_paths, IMAGES_PER_FOLDER, preprocess_func, IMAGE_SIZE)
        class_features = model.predict(images, batch_size=32, verbose=1)
        features.append(class_features)
        labels.extend([label] * IMAGES_PER_FOLDER)

    # Save extracted features
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(os.path.join(OUTPUT_PATH, f"{model_name}_features.pkl"), "wb") as f:
        pickle.dump(np.vstack(features), f)
    with open(os.path.join(OUTPUT_PATH, "labels.pkl"), "wb") as f:
        pickle.dump(np.array(labels), f)

print("Feature extraction completed.")

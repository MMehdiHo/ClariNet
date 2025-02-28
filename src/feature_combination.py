import os
import numpy as np
from config import OUTPUT_PATH

# Load features
features = {
    "vgg16": np.load(os.path.join(OUTPUT_PATH, "vgg16_features.npy")),
    "resnet50": np.load(os.path.join(OUTPUT_PATH, "resnet50_features.npy")),
    "inception": np.load(os.path.join(OUTPUT_PATH, "inception_features.npy")),
    "vit": np.load(os.path.join(OUTPUT_PATH, "vit_features.npy")),
}

# Feature combinations
feature_combinations = {
    "vgg16+vit": np.hstack([features["vgg16"], features["vit"]]),
    "resnet50+vit": np.hstack([features["resnet50"], features["vit"]]),
    "inception+vit": np.hstack([features["inception"], features["vit"]]),
    "all_cnn+vit": np.hstack([features["vgg16"], features["resnet50"], features["inception"], features["vit"]]),
}

# Save combined features
for name, combined in feature_combinations.items():
    np.save(os.path.join(OUTPUT_PATH, f"{name}_features.npy"), combined)

print("Feature combination completed.")

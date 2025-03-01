# Lung Cancer Classification using CNN models and Vision Transformers (ViTs)

## 📌 Overview
This project extracts features from histopathological images using pre-trained CNN models (such as Vgg16, Vgg19, Resnet 50, Resnet 101,...) and trains an ensemble classifier for lung cancer classification.

## 🔑 Key Contributions
- Hybrid Feature Extraction: Combines CNN-based feature maps with ViT embeddings for robust representation.
- Automated Model Combination: Evaluates different feature fusion strategies for improved classification.
- Multi-Class Tumor Classification: Designed for distinguishing between tumor, stroma, and normal tissue.

---

## 🏗 Framework Architecture
The framework consists of three main components:

1️⃣ Feature Extraction  
   - CNN-Based Extractors: Uses models like VGG16, ResNet50, and EfficientNetB0 to obtain spatial features.  
   - ViT-Based Extractor: Leverages Vision Transformers (ViT) for capturing long-range dependencies.

2️⃣ Feature Combination  
   - Merges extracted features from different models.  
   - Evaluates all possible feature combinations for optimal classification.

3️⃣ Classification Model  
   - Fully connected neural network (MLP) for multi-class classification.  
   - Trained with Adam optimizer, binary cross-entropy loss, and dropout for regularization.

### 📊 Framework Architecture Diagram:
The following diagram illustrates the overall framework of our method, including data acquisition, feature extraction, and classification.

![Framework Architecture](https://github.com/MMehdiHo/ClariNet/blob/main/data/image/method.png)

The framework consists of three main components:
1. Dataset Balancing
2. Feature Extraction
3. Ensemble Classification

---

## 📊 Results
We evaluated our method on two distinct histopathology datasets:

- [WSSS4LUAD](https://wsss4luad.grand-challenge.org/WSSS4LUAD/)  
- [DHMC-LUAD](https://bmirds.github.io/LungCancer/)  

### 🔥 Performance:
- Accuracy: 93.0% (±2.5) | Precision: 0.9160 (±1.5)  
- Accuracy: 95.1% (±0.7) | Precision: 95.4% (±0.6)  

---
# Data Balancing Script

This script balances a dataset by selecting images from different folders.

## 📌 How to Use

1. Install dependencies:
2. Modify `config.py` to set correct file paths.
3. Run the script:


## 📂 Project Structure
- `src/feature_extraction.py`: Loads images, extracts CNN features.
- `src/model_training.py`: Builds and trains the ensemble classifier.
- `data/`: Contains sample dataset files (not all included in the repository).
- `output/`: Stores extracted features and trained models.

## 🛠 Installation
1. Clone the repository:
2. Install dependencies:

## 🚀 Usage
Run the feature extraction script:
python src/feature_extraction.py

Train the model:
python src/model_training.py

## 📜 License
This project is licensed under the MIT License.
---
## 📜 Citation
If you find this work useful, please cite our paper:

```bibtex
@article{hosseini2025clarinet,
  title={ClariNet: Clarifying Histopathological Subtypes with Fuzzy Coverage Deep Ensemble Learning},
  author={Mohammad Mehdi Hosseini and Meghdad Sabouri Rad and Junze Huang and Rakesh Choudhary and Harmen Seizen and Ola El-Zammar and Saverio J. Carello and Michel Nasr and Bardia Yousefi Rodd},
  journal={Under Review},
  year={2025}
}


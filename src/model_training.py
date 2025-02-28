import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from utils import plot_training_history

# Constants
NUM_CLASSES = 5
EPOCHS = 60
BATCH_SIZE = 32
OUTPUT_PATH = "../output/"

# Load extracted features
with open(os.path.join(OUTPUT_PATH, "vgg16_features.pkl"), "rb") as f:
    vgg16_features = pickle.load(f)
with open(os.path.join(OUTPUT_PATH, "resnet50_features.pkl"), "rb") as f:
    resnet50_features = pickle.load(f)
with open(os.path.join(OUTPUT_PATH, "inception_features.pkl"), "rb") as f:
    inception_features = pickle.load(f)
with open(os.path.join(OUTPUT_PATH, "labels.pkl"), "rb") as f:
    labels = pickle.load(f)

# Combine features
combined_features = np.hstack([vgg16_features, resnet50_features, inception_features])
labels = to_categorical(labels, NUM_CLASSES)

# Split data
X_train, X_val, y_train, y_val = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

# Define Model
def build_classifier(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train Model
model = build_classifier(combined_features.shape[1:])
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

# Evaluate Model
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Plot training history
plot_training_history(history)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

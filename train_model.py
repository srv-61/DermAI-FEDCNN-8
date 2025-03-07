import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB6, ResNet152
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (224, 224)  
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 5  

# Define paths
TRAIN_DIR = "D:/SOHAN/8TH SEM/Capstone Project Phase 2/CNN & FED LEARNING/Datasets/Train Data"
VAL_DIR = "D:/SOHAN/8TH SEM/Capstone Project Phase 2/CNN & FED LEARNING/Datasets/Validation Data"

#  Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

#  Load EfficientNetB6
efficientnet_b6_model = EfficientNetB6(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze most layers, but unfreeze the last 30 layers for fine-tuning
for layer in efficientnet_b6_model.layers[:-30]:  
    layer.trainable = False

#  Load ResNet152
resnet152_model = ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze only the last 50 layers for fine-tuning
for layer in resnet152_model.layers[:-50]:  
    layer.trainable = False

#  Build EfficientNetB6 Model
efficientnet_b6 = Sequential([
    efficientnet_b6_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

#  Build ResNet152 Model
resnet152 = Sequential([
    resnet152_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

#  **Fixed Optimizer (Using Standard Float LR Instead of CosineDecay)**
efficientnet_b6.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

resnet152.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

#  Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, min_lr=1e-7), 
    ModelCheckpoint("efficientnet_b6_model.keras", save_best_only=True),
    ModelCheckpoint("resnet152_model.keras", save_best_only=True)
]

# Train EfficientNetB6
efficientnet_b6_history = efficientnet_b6.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Train ResNet152
resnet152_history = resnet152.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Load Best Weights
efficientnet_b6.load_weights("efficientnet_b6_model.keras")
resnet152.load_weights("resnet152_model.keras")

# Make Predictions
efficientnet_b6_preds = efficientnet_b6.predict(validation_generator)
resnet152_preds = resnet152.predict(validation_generator)

# Weighted Averaging
combined_preds = (0.6 * efficientnet_b6_preds) + (0.4 * resnet152_preds)

# Get True Labels
true_labels = validation_generator.classes

# Get Final Predicted Classes
predicted_labels = np.argmax(combined_preds, axis=1)

# Compute Accuracy & Precision
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')

print(f"Ensemble Model Accuracy: {accuracy:.4f}")
print(f"Ensemble Model Precision: {precision:.4f}")

# Plot Training History
def plot_training_history(efficientnet_b6_history, resnet152_history):
    plt.figure(figsize=(12, 6))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(efficientnet_b6_history.history['accuracy'], label='EfficientNetB6 Train Accuracy')
    plt.plot(efficientnet_b6_history.history['val_accuracy'], label='EfficientNetB6 Val Accuracy')
    plt.plot(resnet152_history.history['accuracy'], label='ResNet152 Train Accuracy')
    plt.plot(resnet152_history.history['val_accuracy'], label='ResNet152 Val Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(efficientnet_b6_history.history['loss'], label='EfficientNetB6 Train Loss')
    plt.plot(efficientnet_b6_history.history['val_loss'], label='EfficientNetB6 Val Loss')
    plt.plot(resnet152_history.history['loss'], label='ResNet152 Train Loss')
    plt.plot(resnet152_history.history['val_loss'], label='ResNet152 Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the Plotting Function
plot_training_history(efficientnet_b6_history, resnet152_history)

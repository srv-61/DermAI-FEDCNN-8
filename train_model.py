import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB6, ResNet152
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (224, 224)  # Image size set to 224x224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 5  # Number of dermatological diseases

# Define paths
TRAIN_DIR = "D:/SOHAN/8TH SEM/Capstone Project Phase 2/CNN & FED LEARNING/Datasets/Train Data"
VAL_DIR = "D:/SOHAN/8TH SEM/Capstone Project Phase 2/CNN & FED LEARNING/Datasets/Validation Data"

# Load EfficientNetB6 without top layers
efficientnet_b6_model = EfficientNetB6(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
efficientnet_b6_model.trainable = False  # Freeze all layers

# Load ResNet152 without top layers
resnet152_model = ResNet152(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# **Unfreeze last 50 layers of ResNet-152 for fine-tuning**
for layer in resnet152_model.layers[-50:]:
    layer.trainable = True

# Create EfficientNetB6 model
efficientnet_b6 = Sequential([
    efficientnet_b6_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Create ResNet152 model
resnet152 = Sequential([
    resnet152_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Use AdamW optimizer with weight decay for stability
efficientnet_b6.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5), 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])

resnet152.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-5), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])  # Lower learning rate for fine-tuned layers

# Set up ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255)
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

# Initialize variables to store history
efficientnet_b6_history = None
resnet152_history = None

# Train EfficientNetB6
efficientnet_b6_model_path = "efficientnet_b6_model.keras"
efficientnet_b6_history = efficientnet_b6.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[ModelCheckpoint(efficientnet_b6_model_path, save_best_only=True)]
)

# Train ResNet152
resnet152_model_path = "resnet152_model.keras"
resnet152_history = resnet152.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[ModelCheckpoint(resnet152_model_path, save_best_only=True)]
)

# Load the best weights of both models
efficientnet_b6.load_weights(efficientnet_b6_model_path)
resnet152.load_weights(resnet152_model_path)

# Make predictions
efficientnet_b6_preds = efficientnet_b6.predict(validation_generator)
resnet152_preds = resnet152.predict(validation_generator)

# Combine predictions (averaging probabilities)
combined_preds = (efficientnet_b6_preds + resnet152_preds) / 2

# Get the true labels
true_labels = validation_generator.classes

# Find the predicted class
predicted_labels = np.argmax(combined_preds, axis=1)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Combined Accuracy: {accuracy:.4f}")

# Calculate precision
precision = precision_score(true_labels, predicted_labels, average='macro')
print(f"Combined Precision: {precision:.4f}")

# Plot training and validation accuracy/loss
def plot_training_history(efficientnet_b6_history, resnet152_history):
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(efficientnet_b6_history.history['accuracy'], label='EfficientNetB6 Train Acc')
    plt.plot(efficientnet_b6_history.history['val_accuracy'], label='EfficientNetB6 Val Acc')
    plt.plot(resnet152_history.history['accuracy'], label='ResNet152 Train Acc')
    plt.plot(resnet152_history.history['val_accuracy'], label='ResNet152 Val Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
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

# Call the plotting function
plot_training_history(efficientnet_b6_history, resnet152_history)

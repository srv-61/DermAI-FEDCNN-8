import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB6, ResNet152
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (224, 224)  
BATCH_SIZE = 16
EPOCHS = 50  
NUM_CLASSES = 5  

# Paths to datasets
TRAIN_DIR = "D:/SOHAN/8TH SEM/Capstone Project Phase 2/CNN & FED LEARNING/Datasets/Train Data"
VAL_DIR = "D:/SOHAN/8TH SEM/Capstone Project Phase 2/CNN & FED LEARNING/Datasets/Validation Data"

# Load EfficientNet-B6 (without top layers)
efficientnet_b6_model = EfficientNetB6(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
efficientnet_b6_model.trainable = False  

# Load ResNet-152 (without top layers)
resnet152_model = ResNet152(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Unfreeze last 50 layers of ResNet-152 for fine-tuning
for layer in resnet152_model.layers[-50:]:  
    layer.trainable = True

# Define EfficientNet-B6 model
efficientnet_b6 = Sequential([
    efficientnet_b6_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')  
])

# Define ResNet-152 model
resnet152 = Sequential([
    resnet152_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')  
])

# Optimizer (Lower learning rate for fine-tuning)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile models
efficientnet_b6.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
resnet152.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Image data generators
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

# Callbacks to prevent overfitting
callbacks = [
    ModelCheckpoint("efficientnet_b6_model.keras", save_best_only=True),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
]

callbacks_resnet = [
    ModelCheckpoint("resnet152_model.keras", save_best_only=True),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
]

# Train EfficientNet-B6
efficientnet_b6_history = efficientnet_b6.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Train ResNet-152
resnet152_history = resnet152.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks_resnet
)

# Load best weights
efficientnet_b6.load_weights("efficientnet_b6_model.keras")
resnet152.load_weights("resnet152_model.keras")

# Make predictions
efficientnet_b6_preds = efficientnet_b6.predict(validation_generator)
resnet152_preds = resnet152.predict(validation_generator)

# Combine predictions (averaging probabilities)
combined_preds = (efficientnet_b6_preds + resnet152_preds) / 2

# Get true labels
true_labels = validation_generator.classes

# Find predicted classes
predicted_labels = np.argmax(combined_preds, axis=1)

# Calculate accuracy and precision
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')

print(f"Combined Accuracy: {accuracy:.4f}")
print(f"Combined Precision: {precision:.4f}")

# Plot training and validation results
def plot_training_history(efficientnet_b6_history, resnet152_history):
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(efficientnet_b6_history.history['accuracy'], label='EfficientNet-B6 Train Accuracy')
    plt.plot(efficientnet_b6_history.history['val_accuracy'], label='EfficientNet-B6 Val Accuracy')
    plt.plot(resnet152_history.history['accuracy'], label='ResNet-152 Train Accuracy')
    plt.plot(resnet152_history.history['val_accuracy'], label='ResNet-152 Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(efficientnet_b6_history.history['loss'], label='EfficientNet-B6 Train Loss')
    plt.plot(efficientnet_b6_history.history['val_loss'], label='EfficientNet-B6 Val Loss')
    plt.plot(resnet152_history.history['loss'], label='ResNet-152 Train Loss')
    plt.plot(resnet152_history.history['val_loss'], label='ResNet-152 Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function to plot results
plot_training_history(efficientnet_b6_history, resnet152_history)

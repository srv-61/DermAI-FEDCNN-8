# client.py (Federated Learning Client)

import requests
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Constants
SERVER_URL = "http://127.0.0.1:5000"  # Change to server IP if running remotely
NUM_CLASSES = 5

def create_model():
    base_model = EfficientNetB6(weights=None, include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

client_model = create_model()
client_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Simulated local training (Replace with real dataset)
def train_local_model():
    print("Training local model...")
    x_train = np.random.rand(10, 224, 224, 3)  # Fake 10 images
    y_train = np.random.randint(0, NUM_CLASSES, 10)
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    client_model.fit(x_train, y_train, epochs=1, verbose=1)  # Train for 1 epoch locally

# Send local model update to server
def send_update():
    weights_hex = pickle.dumps(client_model.get_weights()).hex()
    response = requests.post(f"{SERVER_URL}/send_update", json={"weights": weights_hex})
    print(response.json())

# Get global model from server
def get_global_model():
    response = requests.get(f"{SERVER_URL}/get_global_model")
    global_weights = pickle.loads(bytes.fromhex(response.json()["weights"]))
    client_model.set_weights(global_weights)
    print("Updated local model with global model weights.")

# Main Federated Learning process
if __name__ == "__main__":
    for _ in range(5):  # Simulate multiple FL rounds
        get_global_model()
        train_local_model()
        send_update()
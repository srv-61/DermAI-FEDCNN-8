# server.py (Flask-based Federated Learning Server with API)

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Global Model
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

global_model = create_model()
global_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

clients = {}  # Dictionary to track registered clients

@app.route('/register_client', methods=['POST'])
def register_client():
    data = request.get_json()
    client_id = data.get("client_id")
    if client_id in clients:
        return jsonify({"message": "Client already registered"}), 400
    clients[client_id] = {"status": "registered"}
    return jsonify({"message": "Client registered successfully"})

@app.route('/start_round', methods=['POST'])
def start_round():
    for client in clients:
        clients[client]["status"] = "training"
    return jsonify({"message": "Federated Learning round started"})

@app.route('/send_update', methods=['POST'])
def receive_client_update():
    data = request.get_json()
    client_id = data.get("client_id")
    client_weights = pickle.loads(bytes.fromhex(data['weights']))
    
    # Aggregate using simple averaging (FedAvg)
    global_weights = global_model.get_weights()
    for i in range(len(global_weights)):
        global_weights[i] = np.mean([client_weights[i], global_weights[i]], axis=0)
    
    global_model.set_weights(global_weights)
    clients[client_id]["status"] = "updated"
    return jsonify({"message": "Update received and model updated"})

@app.route('/get_global_model', methods=['GET'])
def send_global_model():
    weights_hex = pickle.dumps(global_model.get_weights()).hex()
    return jsonify({"weights": weights_hex})

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify(clients)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

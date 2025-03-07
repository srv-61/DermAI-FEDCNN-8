from flask import Flask, request, jsonify, send_file
import os
import torch
from model import get_model

app = Flask(__name__)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
client_updates = []
model_versions = {"resnet152": 0, "efficientnet-b6": 0}

def save_initial_model(model_type):
    """Save an initial global model if it doesn't exist."""
    path = os.path.join(MODEL_DIR, f"global_model_{model_type}.pth")
    if not os.path.exists(path):
        torch.save(get_model(model_type).state_dict(), path)
        print(f"💾 Initial global model ({model_type}) saved.")
    else:
        print(f"✅ Global model ({model_type}) already exists.")

save_initial_model("resnet152")
save_initial_model("efficientnet-b6")

@app.route("/get_model_version", methods=["GET"])
def get_model_version():
    """Clients check for the latest model version before downloading."""
    model_type = request.args.get("model_type")
    return jsonify({"model_type": model_type, "version": model_versions[model_type]})

@app.route("/get_global_model", methods=["GET"])
def get_global_model():
    """Serve the latest global model to clients."""
    model_type = request.args.get("model_type")
    model_path = os.path.join(MODEL_DIR, f"global_model_{model_type}.pth")
    
    if not os.path.exists(model_path):
        return jsonify({"error": f"Global model {model_type} not found"}), 404
    
    return send_file(model_path, as_attachment=True)

@app.route("/submit_update", methods=["POST"])
def receive_update():
    """Receive and store model updates from clients."""
    try:
        client_id = request.form.get("client_id")
        model_type = request.form.get("model_type")
        model_file = request.files.get("model")
        
        if not client_id or not model_type or not model_file:
            return jsonify({"error": "Missing client_id, model_type, or model file"}), 400
        
        model_path = os.path.join(MODEL_DIR, f"client_{client_id}_{model_type}.pth")
        model_file.save(model_path)
        client_updates.append((client_id, model_path, model_type))
        print(f"📥 Received update from Client {client_id}")

        if len(client_updates) >= 3:
            aggregate_updates()
        
        return jsonify({"message": "Model update received"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def aggregate_updates():
    """Aggregate client model updates and update the global model."""
    global client_updates
    if not client_updates:
        print("⚠️ No updates to aggregate.")
        return
    
    model_type = client_updates[0][2]
    global_model_path = os.path.join(MODEL_DIR, f"global_model_{model_type}.pth")
    
    if not os.path.exists(global_model_path):
        print(f"❌ Global model {model_type} not found. Skipping aggregation.")
        return
    
    global_model = get_model(model_type)
    global_model.load_state_dict(torch.load(global_model_path, map_location="cpu"), strict=False)
    
    num_clients = len(client_updates)
    model_states = [torch.load(path, map_location="cpu") for _, path, _ in client_updates]
    
    global_state_dict = global_model.state_dict()
    for key in global_state_dict.keys():
        global_state_dict[key] = sum(model.get(key, torch.zeros_like(global_state_dict[key])) for model in model_states) / num_clients
    
    global_model.load_state_dict(global_state_dict, strict=False)
    torch.save(global_model.state_dict(), global_model_path)
    model_versions[model_type] += 1
    
    print(f"🔄 Aggregated {num_clients} client updates. New version: {model_versions[model_type]}")
    client_updates = []

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)

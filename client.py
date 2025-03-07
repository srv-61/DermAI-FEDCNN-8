import os
import torch
import requests
from model import get_model
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

SERVER_URL = "http://127.0.0.1:10000"

def get_user_input():
    client_id = input("Enter Client ID: ").strip()
    model_type = input("Choose Model (resnet152/efficientnet-b6): ").strip().lower()
    while model_type not in ["resnet152", "efficientnet-b6"]:
        print("❌ Invalid choice. Choose either 'resnet152' or 'efficientnet-b6'.")
        model_type = input("Choose Model (resnet152/efficientnet-b6): ").strip().lower()
    return client_id, model_type

def get_model_version(model_type):
    """Check the latest global model version from the server."""
    response = requests.get(f"{SERVER_URL}/get_model_version", params={"model_type": model_type})
    if response.status_code == 200:
        return response.json().get("version", 0)
    return 0

def download_global_model(client_id, model_type):
    response = requests.get(f"{SERVER_URL}/get_global_model", params={"model_type": model_type})
    
    if response.status_code == 200 and response.content:
        model_path = f"global_model_{model_type}.pth"
        with open(model_path, "wb") as f:
            f.write(response.content)
        print(f"⬇️ Client {client_id} downloaded global model ({model_type}).")
        return model_path
    else:
        print(f"❌ Failed to download global model. Server response: {response.text}")
        exit(1)

def train_local_model(client_id, model_type, global_model_path):
    model = get_model(model_type)
    model.load_state_dict(torch.load(global_model_path, map_location="cpu"), strict=False)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.randn(16, 3, 224, 224), torch.randint(0, 10, (16,)))
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=data_loader, 
        target_epsilon=3.0, target_delta=1e-5, max_grad_norm=1.2, epochs=1
    )

    model.train()
    for epoch in range(1):
        for batch_input, batch_target in data_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_input), batch_target)
            loss.backward()
            optimizer.step()

    print(f"🛡️ Client {client_id} trained using DP (ε=3.0). Loss: {loss.item()}")
    
    local_model_path = f"client_model_{client_id}_{model_type}.pth"
    torch.save(model.state_dict(), local_model_path)
    return local_model_path

def send_update(client_id, model_type, model_path):
    with open(model_path, "rb") as f:
        response = requests.post(f"{SERVER_URL}/submit_update", files={"model": f}, data={"client_id": client_id, "model_type": model_type})
    
    if response.status_code == 200:
        print(f"✅ Client {client_id} sent model update ({model_type}) successfully.")
        
        # After sending update, check if there's a new global model
        server_version = get_model_version(model_type)
        if server_version > 0:
            download_global_model(client_id, model_type)
    else:
        print(f"❌ Failed to send update: {response.text}")

if __name__ == "__main__":
    client_id, model_type = get_user_input()
    global_model_path = download_global_model(client_id, model_type)
    local_model_path = train_local_model(client_id, model_type, global_model_path)
    send_update(client_id, model_type, local_model_path)

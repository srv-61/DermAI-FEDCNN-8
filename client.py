import flwr as fl
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b6, resnet152

# ----- 1️⃣ DEVICE CONFIGURATION -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- 2️⃣ HYPERPARAMETER SETTINGS -----
BATCH_SIZE = 32  # Adjust based on system capacity
LEARNING_RATE = 0.01  # Adjust for better convergence

# ----- 3️⃣ LOAD CLIENT DATA -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    root="D:/SOHAN/8TH SEM/Capstone Project Phase 2/CNN & FED LEARNING/Datasets/Train Data",
    transform=transform
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----- 4️⃣ DEFINE MODEL (EfficientNet-B6 + ResNet-152) -----
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()

        # EfficientNet-B6
        self.efficient_net = efficientnet_b6(weights="IMAGENET1K_V1")  # Updated syntax
        self.efficient_net.classifier[1] = nn.Linear(self.efficient_net.classifier[1].in_features, 512)

        # ResNet-152
        self.resnet = resnet152(weights="IMAGENET1K_V1")  # Updated syntax
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)

        # Final Classification Layer
        self.fc = nn.Linear(1024, len(train_dataset.classes))

    def forward(self, x):
        eff_out = self.efficient_net(x)
        res_out = self.resnet(x)
        combined = torch.cat((eff_out, res_out), dim=1)  # Concatenating outputs
        return self.fc(combined)

# Initialize model and move to GPU (if available)
model = CombinedModel().to(device)

# ----- 5️⃣ TRAIN FUNCTION -----
def train():
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Defined inside train()

    for epoch in range(5):  # Train for 5 epochs
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move to GPU

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/5] Loss: {loss.item():.4f}")

# ----- 6️⃣ FL CLIENT IMPLEMENTATION -----
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def fit(self, parameters, config):
        model.load_state_dict({k: torch.tensor(v).to(device) for k, v in zip(model.state_dict().keys(), parameters)})
        train()
        return self.get_parameters(config), len(train_dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(train_dataset), {}

# Start Federated Learning Client
if __name__ == "__main__":
    fl.client.start_client(
        server_address="192.168.1.8:9091",
        client=FlowerClient().to_client(),
    )

# AI Credit: Code written with assistance from ChatGPT. 
import torch
import torch.nn as nn
from torchviz import make_dot

# Define the improved model class
class ImprovedJokeRegressor(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedJokeRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 1)
        self.initialize_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x.squeeze()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and move it to the appropriate device
input_dim = 768
model = ImprovedJokeRegressor(input_dim).to(device)

# Switch to evaluation mode to avoid batch size issues with BatchNorm
model.eval()

# Create a dummy input tensor and move it to the same device as the model
dummy_input = torch.randn(1, input_dim).to(device)

# Create a graph of the model
model_graph = make_dot(model(dummy_input), params=dict(model.named_parameters()))

# Save the graph to a file
model_graph.render("improved_joke_regressor", format="png")

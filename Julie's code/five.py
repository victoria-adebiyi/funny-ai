import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Define the dataset class
class JokesDataset(Dataset):
    def __init__(self, jokes, scores):
        self.jokes = jokes
        self.scores = scores

    def __len__(self):
        return len(self.jokes)

    def __getitem__(self, idx):
        joke = self.jokes[idx]
        score = torch.tensor(self.scores[idx], dtype=torch.float32)
        return {
            'joke': joke,
            'score': score
        }

class ImprovedJokeRegressor(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedJokeRegressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(128 * (input_dim // 2), 1024)  # Adjust dimensions based on pooling and conv layers
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
        x = x.transpose(1, 2)  # Change shape to (batch_size, input_dim, seq_length)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output from conv layers
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
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# Load and preprocess data
def load_data(train_path, test_path):
    # Read files with the correct delimiter
    train_df = pd.read_csv(train_path, sep=';;;;;', engine='python', header=0, usecols=[0, 1], names=['score', 'joke'])
    test_df = pd.read_csv(test_path, sep=';;;;;', engine='python', header=0, usecols=[0, 1], names=['score', 'joke'])
    
    # Extract jokes and scores
    train_jokes = train_df['joke'].tolist()
    train_scores = train_df['score'].tolist()
    test_jokes = test_df['joke'].tolist()
    test_scores = test_df['score'].tolist()

    return train_jokes, train_scores, test_jokes, test_scores

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if score < self.best_score - self.delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Main training function
def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    for batch in train_loader:
        jokes = batch['joke']
        scores = batch['score'].to(device)

        # Ensure jokes are strings
        if not all(isinstance(joke, str) for joke in jokes):
            raise ValueError("All jokes should be strings")

        # Encode jokes
        embeddings = sbert_model.encode(jokes, convert_to_tensor=False)
        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        
        predictions = model(embeddings)

        loss = criterion(predictions, scores)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Main evaluation function
def evaluate_model(val_loader, model, criterion, device):
    model.eval()
    val_loss = 0
    total_mae = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            jokes = batch['joke']
            scores = batch['score'].to(device)

            # Ensure jokes are strings
            if not all(isinstance(joke, str) for joke in jokes):
                raise ValueError("All jokes should be strings")

            # Encode jokes
            embeddings = sbert_model.encode(jokes, convert_to_tensor=False)
            embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)

            predictions = model(embeddings)

            loss = criterion(predictions, scores)
            mae = mean_absolute_error(scores.cpu(), predictions.cpu())
            val_loss += loss.item()
            total_mae += mae

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(scores.cpu().numpy())

    avg_loss = val_loss / len(val_loader)
    avg_mae = total_mae / len(val_loader)
    r2 = r2_score(all_targets, all_predictions)

    return avg_loss, avg_mae, r2

# Main script
if __name__ == "__main__":
    # Load dataset
    train_jokes, train_scores, test_jokes, test_scores = load_data('train.ssv', 'test.ssv')

    # Initialize SBERT model
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Create datasets and dataloaders
    train_dataset = JokesDataset(train_jokes, train_scores)
    test_dataset = JokesDataset(test_jokes, test_scores)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the improved regression model
    joke_regressor = ImprovedJokeRegressor(input_dim=768)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(joke_regressor.parameters(), lr=1e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    joke_regressor.to(device)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    num_epochs = 50
    for epoch in range(num_epochs):
        train_model(train_loader, joke_regressor, criterion, optimizer, device)
        avg_loss, avg_mae, r2 = evaluate_model(test_loader, joke_regressor, criterion, device)
        print(f'Epoch {epoch+1}, Validation Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, R2 Score: {r2:.4f}')

        # Early stopping check
        early_stopping(avg_mae)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save the model
    torch.save(joke_regressor.state_dict(), 'joke_regressor.pth')
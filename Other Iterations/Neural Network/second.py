# AI Credit: Code written with assistance from ChatGPT. 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

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

# Define the improved model class
class ImprovedJokeRegressor(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedJokeRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)  # Added Batch Normalization
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)  # Added Batch Normalization
        self.dropout = nn.Dropout(0.3)  # Reduced dropout rate
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Apply Batch Normalization
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)  # Apply Batch Normalization
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()  # Remove the extra dimension

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
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
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
        embeddings = sbert_model.encode(jokes, convert_to_tensor=False)  # Get numpy arrays
        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)  # Convert numpy arrays to tensors
        
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
    with torch.no_grad():
        for batch in val_loader:
            jokes = batch['joke']
            scores = batch['score'].to(device)

            # Ensure jokes are strings
            if not all(isinstance(joke, str) for joke in jokes):
                raise ValueError("All jokes should be strings")

            # Encode jokes
            embeddings = sbert_model.encode(jokes, convert_to_tensor=False)  # Get numpy arrays
            embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)  # Convert numpy arrays to tensors

            predictions = model(embeddings)

            loss = criterion(predictions, scores)
            mae = calculate_mae(predictions, scores)
            val_loss += loss.item()
            total_mae += mae.item()

    avg_loss = val_loss / len(val_loader)
    avg_mae = total_mae / len(val_loader)
    return avg_loss, avg_mae

# Calculate MAE
def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

# Main script
if __name__ == "__main__":
    # Load dataset
    train_jokes, train_scores, test_jokes, test_scores = load_data('train.ssv', 'test.ssv')

    # Initialize SBERT model
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Create datasets and dataloaders
    train_dataset = JokesDataset(train_jokes, train_scores)
    test_dataset = JokesDataset(test_jokes, test_scores)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Increased batch size
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Increased batch size

    # Initialize the improved regression model
    joke_regressor = ImprovedJokeRegressor(input_dim=768)  # 768 is the dimension of BERT embeddings
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(joke_regressor.parameters(), lr=1e-4)  # Using AdamW optimizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    joke_regressor.to(device)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    num_epochs = 50  # Increased number of epochs
    for epoch in range(num_epochs):
        train_model(train_loader, joke_regressor, criterion, optimizer, device)
        avg_loss, avg_mae = evaluate_model(test_loader, joke_regressor, criterion, device)
        print(f'Epoch {epoch+1}, Validation Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}')

        # Early stopping check
        early_stopping(avg_mae)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save the model
    torch.save(joke_regressor.state_dict(), 'joke_regressor.pth')
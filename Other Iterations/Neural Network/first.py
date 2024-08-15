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

# Define the model class
class JokeRegressor(nn.Module):
    def __init__(self, input_dim):
        super(JokeRegressor, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze()  # Remove the extra dimension

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
            val_loss += loss.item()

    return val_loss / len(val_loader)

# Main script
if __name__ == "__main__":
    # Load dataset
    train_jokes, train_scores, test_jokes, test_scores = load_data('train.ssv', 'test.ssv')

    # Initialize SBERT model
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Create datasets and dataloaders
    train_dataset = JokesDataset(train_jokes, train_scores)
    test_dataset = JokesDataset(test_jokes, test_scores)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize the regression model
    joke_regressor = JokeRegressor(input_dim=768)  # 768 is the dimension of BERT embeddings
    criterion = nn.MSELoss()
    optimizer = optim.Adam(joke_regressor.parameters(), lr=1e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    joke_regressor.to(device)

    # Train the model
    for epoch in range(10):
        train_model(train_loader, joke_regressor, criterion, optimizer, device)
        val_loss = evaluate_model(test_loader, joke_regressor, criterion, device)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')

    # Save the model
    torch.save(joke_regressor.state_dict(), 'joke_regressor.pth')

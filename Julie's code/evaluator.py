import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
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

# Define the evaluation function
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
    rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_targets))**2))
    pearson_corr, _ = pearsonr(all_targets, all_predictions)
    spearman_corr, _ = spearmanr(all_targets, all_predictions)

    return avg_loss, avg_mae, r2, rmse, pearson_corr, spearman_corr

# Load the saved model
model_path = 'joke_regressor2.pth'
input_dim = 768
model = ImprovedJokeRegressor(input_dim=input_dim)
model.load_state_dict(torch.load(model_path))

# Load the test data
test_path = 'test.ssv'
test_df = pd.read_csv(test_path, sep=';;;;;', engine='python', header=0, usecols=[0, 1], names=['score', 'joke'])
test_jokes = test_df['joke'].tolist()
test_scores = test_df['score'].tolist()

# Initialize SBERT model for encoding jokes
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Create the dataset and dataloader
test_dataset = JokesDataset(test_jokes, test_scores)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.MSELoss()

# Calculate metrics
avg_loss, avg_mae, r2, rmse, pearson_corr, spearman_corr = evaluate_model(test_loader, model, criterion, device)

# Print the results
print(f'Validation Loss: {avg_loss:.4f}')
print(f'Mean Absolute Error (MAE): {avg_mae:.4f}')
print(f'R² Score: {r2:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Pearson Correlation: {pearson_corr:.4f}')
print(f'Spearman Correlation: {spearman_corr:.4f}')
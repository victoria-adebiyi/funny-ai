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
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()

# Load the model and test data
def load_model_and_data(model_path, test_path):
    # Load the model
    input_dim = 768
    model = ImprovedJokeRegressor(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))

    # Load the test data
    test_df = pd.read_csv(test_path, sep=';;;;;', engine='python', header=0, usecols=[0, 1], names=['score', 'joke'])
    test_jokes = test_df['joke'].tolist()
    test_scores = test_df['score'].tolist()

    return model, test_jokes, test_scores

# Evaluate the model
def evaluate_model(model, test_jokes, test_scores, device):
    model.eval()
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    test_embeddings = sbert_model.encode(test_jokes, convert_to_tensor=False)
    test_embeddings = torch.tensor(test_embeddings, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(test_embeddings).cpu().numpy()

    # Calculate evaluation metrics
    mae = mean_absolute_error(test_scores, predictions)
    r2 = r2_score(test_scores, predictions)
    rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_targets))**2))
    pearson_corr, _ = pearsonr(test_scores, predictions)
    spearman_corr, _ = spearmanr(test_scores, predictions)

    return mae, r2, rmse, pearson_corr, spearman_corr

# Main script
if __name__ == "__main__":
    model_path = 'joke_regressor2.pth'
    test_path = 'test.ssv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model and data
    model, test_jokes, test_scores = load_model_and_data(model_path, test_path)
    model.to(device)

    # Evaluate the model
    mae, r2, rmse, pearson_corr, spearman_corr = evaluate_model(model, test_jokes, test_scores, device)

    # Print the results
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'R² Score: {r2:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Pearson Correlation: {pearson_corr:.4f}')
    print(f'Spearman Rank Correlation: {spearman_corr:.4f}')
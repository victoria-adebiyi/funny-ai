import torch
import argparse
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from jokerater import predict_joke_score
from scoreconvert import reddit_score_to_10_scale
import warnings
warnings.filterwarnings("ignore")

# Constants
MODEL_PATH = 'joke_regressor.pth'
INPUT_DIM = 768

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

def load_joke_from_file(filename):
    """Load the joke from a text file."""
    with open(filename, 'r') as file:
        return file.read().strip()

def main(joke_file, raw_reddit_score):
    # Load the joke from the specified text file
    joke = load_joke_from_file(joke_file)
    
    # Initialize SBERT model for encoding jokes
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    # Load the trained joke regressor model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedJokeRegressor(input_dim=INPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    model.to(device)
    model.eval()
    
    # Predict the joke score
    predicted_score = predict_joke_score(joke, model, sbert_model, device)
    
    # Convert raw Reddit score to a 0-10 scale
    scaled_reddit_score = reddit_score_to_10_scale(raw_reddit_score)
    
    # Calculate the difference
    score_difference = predicted_score - scaled_reddit_score
    
    # Output the results
    print(f"Predicted Joke Score: {predicted_score:.2f}")
    print(f"Scaled Reddit Score: {scaled_reddit_score:.2f}")
    print(f"Difference between the scores: {abs(score_difference):.2f}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Compare Reddit score with joke prediction.")
    parser.add_argument('joke_file', type=str, help='The path to the file containing the joke.')
    parser.add_argument('raw_reddit_score', type=float, help='The raw Reddit score to be converted and compared.')
    
    args = parser.parse_args()
    
    main(args.joke_file, args.raw_reddit_score)
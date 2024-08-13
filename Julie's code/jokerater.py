import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

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

# Load the saved model
model_path = 'joke_regressor.pth'
input_dim = 768
model = ImprovedJokeRegressor(input_dim=input_dim)
model.load_state_dict(torch.load(model_path))

# Initialize SBERT model for encoding jokes
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Function to predict the score for a new joke
def predict_joke_score(joke):
    # Encode the joke using SBERT
    embedding = sbert_model.encode([joke], convert_to_tensor=True)
    embedding = embedding.to(device)
    
    # Predict the score
    with torch.no_grad():
        score = model(embedding).item()
    
    # Convert the score to an integer
    return score

# Input a new joke
new_joke = 'Barack Obama walks into a bar, but he is invisible. After attracting the bartender’s attention, the bartender says "Ok, I\'ll bite. Why are you invisible?" Barack says "Well, I found a bottle on the beach and...then I rubbed it." "And then...importantly...A genie came out." "The genie said I could have...3 wishes." For my first wish, I said "Let me say this, and this is profoundly important...I want Michelle to marry me...I love her,...and I think America will love her too." That wish was granted. For my second wish, I said "Like all patriotic Americans, I am deeply patriotic...and I want to be President...of the United States...so I can serve my country." That wish was granted too. And then, for my third wish, I started by saying "Let me be clear..."'

# Predict the score
predicted_score = predict_joke_score(new_joke)

# Print the result
print(f"Predicted Score for the joke: {predicted_score}")
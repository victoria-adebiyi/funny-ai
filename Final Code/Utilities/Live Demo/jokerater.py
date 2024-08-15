# AI Credit: Code written with assistance from ChatGPT.
# Scores jokes using a provided model. 
import torch
from sentence_transformers import SentenceTransformer

# Function to predict the score for a new joke
def predict_joke_score(joke, model, sbert_model, device):
    # Encode the joke using SBERT
    embedding = sbert_model.encode([joke], convert_to_tensor=True)
    embedding = embedding.to(device)
    
    # Predict the score
    with torch.no_grad():
        score = model(embedding).item()
    
    return score

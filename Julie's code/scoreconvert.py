import numpy as np

def reddit_score_to_10_scale(score, max_score=136353):
    """
    Converts a Reddit score (0-136353) to a 0-10 scale using a logarithmic function.
    Scores of 0 remain 0.
    
    Parameters:
    score (int or float): The original Reddit score to be converted.
    max_score (int or float): The maximum Reddit score (default is 136353).

    Returns:
    float: The converted score on a 0-10 scale.
    """
    if score == 0:
        return 0
    else:
        # Apply the logarithmic scaling
        scaled_score = np.log10(score + 1) / np.log10(max_score + 1) * 10
        return scaled_score

# Example usage
reddit_scores = [0, 10, 100, 1000, 50000, 136353]
scaled_scores = [reddit_score_to_10_scale(score) for score in reddit_scores]

print("Original Scores:", reddit_scores)
print("Scaled Scores:", scaled_scores)
print("Scaled score", reddit_score_to_10_scale(11014))
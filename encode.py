from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

print("reading data")

train_data = pd.read_csv("processed/train.ssv", delimiter=";;;;;")
test_data = pd.read_csv("processed/test.ssv", delimiter=";;;;;")

print("data read")

print("encoding training data")

train_encoded = model.encode(train_data['post'])
test_encoded = model.encode(test_data['post'])

print("encoding complete")

print("saving encoding data")

np.savetxt("processed/train-encoded.csv", train_encoded, delimiter=',')
np.savetxt("processed/test-encoded.csv", test_encoded, delimiter=',')

print("encoding data saved")

print("saving score data")

np.savetxt("processed/train-scores.csv", train_data['score'], delimiter=',')
np.savetxt("processed/test-scores.csv", test_data['score'], delimiter=',')

print("score data stored")

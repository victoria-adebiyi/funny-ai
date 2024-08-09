from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error

models = ["lin", "forest"]

print("reading data")

test_encoded = np.loadtxt("processed/test-encoded.csv", delimiter=',')
test_scores = np.loadtxt("processed/test-scores.csv", delimiter=',')

print("data read")

for model in models:

    print("reading " + model + " model")

    regressor = joblib.load('models/' + model + "-model.pkl")

    print(model + " model read")

    print("predicting " + model + " test data")

    predictions = regressor.predict(test_encoded)

    print(model + " test data predicted")

    print("calculating " + model + " mse")

    mse = round(mean_squared_error(test_scores, predictions), 3)

    print(model + " mse is " + str(mse))
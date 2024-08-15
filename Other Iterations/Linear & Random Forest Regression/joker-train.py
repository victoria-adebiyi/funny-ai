# AI Credit: Code written with assistance from ChatGPT. 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

print("reading data")

train_encoded = np.loadtxt("processed/train-encoded.csv", delimiter=',')
train_scores = np.loadtxt("processed/train-scores.csv", delimiter=',')

print("data read")

print("fitting linear regression model")

lin_reg = LinearRegression()
lin_reg.fit(train_encoded, train_scores)

print("linear regression model fit")

print("saving model")

joblib.dump(lin_reg, "models/lin-model.pkl")

print("model saved")

print("fitting random forest regression model")

tree_reg = RandomForestRegressor(random_state=7, n_estimators=250)
tree_reg.fit(train_encoded, train_scores)

print("random forest regression model fit")

print("saving model")

joblib.dump(tree_reg, "models/forest-model.pkl")

print("model saved")
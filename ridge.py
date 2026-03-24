import pandas as pd
from sklearn.dummy import DummyRegressor
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.linear_model import Ridge 
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline


X_train = pd.read_csv("input_data/X_train.csv")
y_train = pd.read_csv("input_data/y_train.csv")
X_test  = pd.read_csv("input_data/X_test.csv")

print(X_train.shape)  # (489, 10001)
print(y_train.shape)  # (489, 1)
print(X_test.shape)   # (200, 10001)

print(X_train.head())
print(X_train.info())
print(y_train.describe())
plt.hist(y_train)
plt.xlabel("Age")
plt.ylabel("Count")
# #plt.show() 

#replace the male with 0,female with 1
X_train["gender"] = X_train["gender"].map({"m": 0, "f": 1})
X_test["gender"] = X_test["gender"].map({"m": 0, "f": 1})
### ridge 
#fitting the model 
model = make_pipeline(StandardScaler(),Ridge(alpha=10))
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)
#create a csv file with the predicted y for the X_test
df = pd.DataFrame(y_pred, columns=["age"])
df.to_csv("y_predict.csv", index=False)
#calculating MSE 
y_pred_train = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred_train) 
print("MSE:", mse) 

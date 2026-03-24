import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.model_selection import cross_val_score

# =========================
# 1. Load data
# =========================
X_train = pd.read_csv("input_data/X_train.csv")
y_train = pd.read_csv("input_data/y_train.csv")
X_test  = pd.read_csv("input_data/X_test.csv")

# Encode gender
X_train["gender"] = X_train["gender"].map({"m": 0, "f": 1})
X_test["gender"] = X_test["gender"].map({"m": 0, "f": 1})

y = y_train.values.ravel()

# =========================
# 2. Preprocessing
# =========================
var_filter = VarianceThreshold(threshold=0.0005)
X_train = var_filter.fit_transform(X_train)
X_test = var_filter.transform(X_test)

# selector = SelectKBest(f_regression, k=700)
# X_train = selector.fit_transform(X_train, y)
# X_test = selector.transform(X_test)

# =========================
# 3. Explicit CV over alphas
# =========================
alphas = np.linspace(0.1, 0.2, 10)

best_alpha = None
best_score = float("inf")

for alpha in alphas:
    model = make_pipeline(
        StandardScaler (),
        Lasso(alpha=alpha, max_iter=20000)
    )
    
    scores = cross_val_score(
        model,
        X_train,
        y,
        cv=5,
        scoring="neg_mean_squared_error"
    )
    
    mse = -scores.mean()
    
    print(f"alpha={alpha:.5f} → MSE={mse:.4f}")
    
    if mse < best_score:
        best_score = mse
        best_alpha = alpha

print("\nBest alpha:", best_alpha)
print("Best CV MSE:", best_score)

# =========================
# 4. Train final model
# =========================
final_model = make_pipeline(
    StandardScaler (),
    Lasso(alpha=best_alpha, max_iter=20000)
)

final_model.fit(X_train, y)

# =========================
# 5. Predict
# =========================
y_pred = final_model.predict(X_test)

# Safety
y_pred = np.nan_to_num(y_pred)
y_pred = np.clip(y_pred, 20, 85)

# Save
df = pd.DataFrame(y_pred, columns=["age"])
df.to_csv("y_pred.csv", index=False)
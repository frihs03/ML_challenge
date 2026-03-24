import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

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
# 2. Remove low variance
# =========================
var_filter = VarianceThreshold(threshold=0.0005)

X_train = var_filter.fit_transform(X_train)
X_test = var_filter.transform(X_test)

# =========================
# 3. Define Lasso models (refined alphas)
# =========================
lasso_0135 = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.135, max_iter=30000, tol=1e-5)
)

lasso_014 = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.14, max_iter=30000, tol=1e-5)
)

lasso_0145 = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.145, max_iter=30000, tol=1e-5)
)

lasso_015 = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.15, max_iter=30000, tol=1e-5)
)

# =========================
# 4. Train models
# =========================
print("Training Lasso models...")

lasso_0135.fit(X_train, y)
lasso_014.fit(X_train, y)
lasso_0145.fit(X_train, y)
lasso_015.fit(X_train, y)

print("Training finished")

# =========================
# 5. Predictions
# =========================
y_0135 = lasso_0135.predict(X_test)
y_014  = lasso_014.predict(X_test)
y_0145 = lasso_0145.predict(X_test)
y_015  = lasso_015.predict(X_test)

# =========================
# 6. Ensemble (optimized weights)
# =========================
y_pred = (
    0.2*y_0135 +
    0.4*y_014 +
    0.3*y_0145 +
    0.1*y_015
)

# =========================
# 7. Safety (minimal)
# =========================
y_pred = np.nan_to_num(y_pred)

# (optional) try with and without clipping
y_pred = np.clip(y_pred, 20, 85)

# =========================
# 8. Save submission
# =========================
df = pd.DataFrame(y_pred, columns=["age"])
df.to_csv("y_pred.csv", index=False)

print("Submission file saved!")

# =========================
# 9. Debug info
# =========================
print("Min age:", y_pred.min())
print("Max age:", y_pred.max())
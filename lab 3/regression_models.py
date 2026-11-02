
# REGRESSION MODELS FOR HOUSE PRICE PREDICTION
# Dataset: California Housing Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------
# 1. LOAD DATASET
# ---------------------------------------

housing = fetch_california_housing()

X = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

y = pd.Series(
    housing.target,
    name="HousePrice"
)

print("California Housing Dataset Loaded!")
print("Total Samples:", X.shape[0])
print("Total Features:", X.shape[1])

print("\nFirst Five Records:")
print(X.head())

# ---------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------

X.fillna(X.median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData Preprocessing Completed!")

# ---------------------------------------
# 3. INITIALIZE REGRESSION MODELS
# ---------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=12,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

results = []
predictions = {}

# ---------------------------------------
# 4. TRAIN AND TEST MODELS
# ---------------------------------------

for name, model in models.items():

    print("\nTraining:", name)

    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    predictions[name] = y_pred

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

# ---------------------------------------
# 5. PERFORMANCE COMPARISON
# ---------------------------------------

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

# Best model based on R2 score
best_model = results_df.loc[
    results_df["R2 Score"].idxmax()
]

print("\nBest Performing Model:")
print(best_model["Model"])

# ---------------------------------------
# 6. ACTUAL VS PREDICTED GRAPH
# ---------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, y_pred) in zip(
    axes, predictions.items()
):

    ax.scatter(
        y_test,
        y_pred,
        alpha=0.3
    )

    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--'
    )

    ax.set_title(name)
    ax.set_xlabel("Actual House Price")
    ax.set_ylabel("Predicted House Price")

plt.tight_layout()
plt.show()

# ---------------------------------------
# 7. R2 SCORE COMPARISON
# ---------------------------------------

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="R2 Score"
)

plt.title("Regression Model Performance Comparison")
plt.ylabel("R2 Score")
plt.xlabel("Regression Models")

plt.tight_layout()
plt.show()

# ---------------------------------------
# 8. RMSE COMPARISON
# ---------------------------------------

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="RMSE"
)

plt.title("RMSE Comparison")
plt.ylabel("RMSE")
plt.xlabel("Regression Models")

plt.tight_layout()
plt.show()

# ---------------------------------------
# 9. SAMPLE PREDICTIONS
# ---------------------------------------

print("\n========== SAMPLE PREDICTIONS ==========")

comparison = pd.DataFrame({
    "Actual Price": y_test.iloc[:10].values,
    "Predicted Price": predictions[
        best_model["Model"]
    ][:10]
})

print(comparison.to_string(index=False))

print("\nProgram Executed Successfully!")

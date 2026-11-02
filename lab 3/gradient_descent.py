
# GRADIENT DESCENT OPTIMIZATION
# Linear Regression using Batch Gradient Descent
# Dataset: California Housing Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

y = housing.target

# Select Median Income feature
X = X[["MedInc"]]

print("California Housing Dataset Loaded!")
print("Total Samples:", len(X))

# ---------------------------------------
# 2. SPLIT DATASET
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------
# 3. FEATURE SCALING
# ---------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to NumPy arrays
X_train = X_train_scaled.flatten()
X_test = X_test_scaled.flatten()

# ---------------------------------------
# 4. GRADIENT DESCENT FUNCTION
# ---------------------------------------

def gradient_descent(X, y, learning_rate, iterations):

    m = len(X)

    # Initialize parameters
    w = 0.0
    b = 0.0

    cost_history = []

    for i in range(iterations):

        # Prediction
        y_pred = w * X + b

        # Calculate errors
        error = y_pred - y

        # Calculate gradients
        dw = (2 / m) * np.sum(error * X)

        db = (2 / m) * np.sum(error)

        # Update parameters
        w = w - learning_rate * dw

        b = b - learning_rate * db

        # Calculate cost
        cost = np.mean((y - (w * X + b)) ** 2)

        cost_history.append(cost)

    return w, b, cost_history


# ---------------------------------------
# 5. TRAIN USING DIFFERENT LEARNING RATES
# ---------------------------------------

learning_rates = [0.001, 0.01, 0.1]

iterations = 1000

results = []

trained_models = {}

for lr in learning_rates:

    print("\nTraining with Learning Rate:", lr)

    w, b, cost_history = gradient_descent(
        X_train,
        y_train,
        lr,
        iterations
    )

    # Predictions
    y_pred = w * X_test + b

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Learning Rate": lr,
        "Weight": w,
        "Bias": b,
        "Final Cost": cost_history[-1],
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[lr] = {
        "weight": w,
        "bias": b,
        "cost": cost_history
    }

    print("Weight:", round(w, 4))
    print("Bias:", round(b, 4))
    print("Final Cost:", round(cost_history[-1], 4))
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))


# ---------------------------------------
# 6. COMPARE RESULTS
# ---------------------------------------

results_df = pd.DataFrame(results)

print("\n========== GRADIENT DESCENT COMPARISON ==========")

print(results_df.to_string(index=False))


# ---------------------------------------
# 7. COST VS ITERATIONS GRAPH
# ---------------------------------------

plt.figure(figsize=(10, 6))

for lr in learning_rates:

    plt.plot(
        trained_models[lr]["cost"],
        label=f"Learning Rate = {lr}"
    )

plt.title("Cost Function vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------
# 8. ACTUAL VS PREDICTED GRAPH
# ---------------------------------------

best_lr = results_df.loc[
    results_df["R2 Score"].idxmax(),
    "Learning Rate"
]

best_w = trained_models[best_lr]["weight"]
best_b = trained_models[best_lr]["bias"]

best_predictions = best_w * X_test + best_b

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.3
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.title(
    f"Actual vs Predicted (Learning Rate = {best_lr})"
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")

plt.tight_layout()
plt.show()


# ---------------------------------------
# 9. REGRESSION LINE VISUALIZATION
# ---------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    y_test,
    alpha=0.3,
    label="Actual Data"
)

sorted_indices = np.argsort(X_test)

plt.plot(
    X_test[sorted_indices],
    best_predictions[sorted_indices],
    color="red",
    label="Gradient Descent Regression Line"
)

plt.title("Linear Regression using Gradient Descent")
plt.xlabel("Standardized Median Income")
plt.ylabel("House Price")
plt.legend()

plt.tight_layout()
plt.show()


# ---------------------------------------
# 10. FINAL RESULT
# ---------------------------------------

print("\n========== FINAL RESULT ==========")

print("Best Learning Rate:", best_lr)

print("Optimized Weight:", round(best_w, 4))

print("Optimized Bias:", round(best_b, 4))

print(
    "Best R2 Score:",
    round(
        results_df.loc[
            results_df["Learning Rate"] == best_lr,
            "R2 Score"
        ].iloc[0],
        4
    )
)

print("\nGradient Descent Executed Successfully!")

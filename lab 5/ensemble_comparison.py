
# LAB 5: ENSEMBLE LEARNING
# Stacking, AdaBoost and Voting
# Comparative Analysis of Classification Models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset and preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Base classifiers
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Ensemble classifiers
from sklearn.ensemble import (
    StackingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix
)

# --------------------------------------------------
# STEP 1: LOAD DATASET
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("\nIRIS DATASET LOADED SUCCESSFULLY")
print("Total Samples:", len(X))
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)

# --------------------------------------------------
# STEP 2: SPLIT DATASET
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# --------------------------------------------------
# STEP 3: DEFINE BASE CLASSIFIERS
# --------------------------------------------------

base_models = [
    (
        "SVM",
        make_pipeline(
            StandardScaler(),
            SVC(
                kernel="linear",
                probability=True,
                random_state=42
            )
        )
    ),

    (
        "KNN",
        make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5)
        )
    ),

    (
        "Decision Tree",
        DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        )
    ),

    (
        "Naive Bayes",
        GaussianNB()
    )
]

# --------------------------------------------------
# STEP 4: DEFINE INDIVIDUAL MODELS
# --------------------------------------------------

models = {
    "SVM": base_models[0][1],

    "KNN": base_models[1][1],

    "Decision Tree": base_models[2][1],

    "Naive Bayes": base_models[3][1]
}

# --------------------------------------------------
# STEP 5: STACKING ENSEMBLE
# --------------------------------------------------

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(
        max_iter=1000
    ),
    cv=5,
    stack_method="predict_proba"
)

# --------------------------------------------------
# STEP 6: ADABOOST ENSEMBLE
# --------------------------------------------------

adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1
    ),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# --------------------------------------------------
# STEP 7: VOTING ENSEMBLE
# --------------------------------------------------

voting = VotingClassifier(
    estimators=base_models,
    voting="soft"
)

# --------------------------------------------------
# STEP 8: ADD ENSEMBLE MODELS
# --------------------------------------------------

models["Stacking"] = stacking
models["AdaBoost"] = adaboost
models["Voting"] = voting

# --------------------------------------------------
# STEP 9: TRAIN AND EVALUATE MODELS
# --------------------------------------------------

results = []

for name, model in models.items():

    print("\n" + "=" * 55)
    print("MODEL:", name)
    print("=" * 55)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Probability predictions
    y_prob = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    loss = log_loss(
        y_test,
        y_prob,
        labels=np.arange(len(iris.target_names))
    )

    # Store results
    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Log Loss": loss
    })

    # Display results
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))
    print("Log Loss:", round(loss, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# STEP 10: COMPARATIVE RESULT TABLE
# --------------------------------------------------

df = pd.DataFrame(results)

print("\n\nFINAL PERFORMANCE COMPARISON")
print("=" * 85)
print(df.round(4).to_string(index=False))

# Save results to CSV
df.to_csv("ensemble_results.csv", index=False)

print("\nResults saved to ensemble_results.csv")

# --------------------------------------------------
# STEP 11: FIND BEST MODEL
# --------------------------------------------------

best_model = df.loc[df["Accuracy"].idxmax()]

print("\nBEST PERFORMING MODEL")
print("Algorithm:", best_model["Algorithm"])
print("Accuracy:", round(best_model["Accuracy"] * 100, 2), "%")

# --------------------------------------------------
# STEP 12: PERFORMANCE COMPARISON GRAPH
# --------------------------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

df.set_index("Algorithm")[metrics].plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1.1)
)

plt.title("Base Models vs Ensemble Models")
plt.xlabel("Classification Algorithms")
plt.ylabel("Performance Score")
plt.xticks(rotation=25)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# STEP 13: LOG LOSS GRAPH
# --------------------------------------------------

plt.figure(figsize=(10, 5))

sns.barplot(
    data=df,
    x="Algorithm",
    y="Log Loss"
)

plt.title("Log Loss Comparison")
plt.xlabel("Classification Algorithms")
plt.ylabel("Log Loss (Lower is Better)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# STEP 14: CONFUSION MATRICES
# --------------------------------------------------

for name, model in models.items():

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )

    plt.title(name + " - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# STEP 15: END
# --------------------------------------------------

print("\n" + "=" * 55)
print("ENSEMBLE LEARNING EXPERIMENT COMPLETED SUCCESSFULLY!")
print("=" * 55)


# Comparative Analysis of Classification Algorithms
# SVM, Decision Tree, KNN and Naive Bayes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report
)

# Step 1: Load dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset Loaded Successfully")
print("Total Samples:", len(X))
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)

# Step 2: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Step 3: Define classification models
models = {
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", probability=True, random_state=42)
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "Naive Bayes": GaussianNB()
}

# Step 4: Train and evaluate models
results = []

for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    # Train model
    model.fit(X_train, y_train)

    # Predict output
    y_pred = model.predict(X_test)

    # Predict probabilities
    y_prob = model.predict_proba(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    loss = log_loss(
        y_test, y_prob,
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

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))
    print("Log Loss:", round(loss, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=iris.target_names,
        zero_division=0
    ))

# Step 5: Display comparative analysis
df = pd.DataFrame(results)

print("\n\nFINAL COMPARISON TABLE")
print(df.round(4).to_string(index=False))

# Step 6: Identify best model
best_model = df.loc[df["Accuracy"].idxmax()]

print("\nBest Performing Algorithm:",
      best_model["Algorithm"])

print("Highest Accuracy:",
      round(best_model["Accuracy"] * 100, 2), "%")

# Step 7: Accuracy, Precision, Recall and F1-score graph
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

df.set_index("Algorithm")[metrics].plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1.1)
)

plt.title("Classification Algorithm Performance")
plt.xlabel("Classification Algorithms")
plt.ylabel("Performance Score")
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Step 8: Log Loss comparison graph
plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="Algorithm",
    y="Log Loss"
)

plt.title("Log Loss Comparison")
plt.xlabel("Classification Algorithms")
plt.ylabel("Log Loss (Lower is Better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Step 9: Confusion matrix visualization
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

print("\nClassification Experiment Completed Successfully!")

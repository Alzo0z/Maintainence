import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)


DATA_PATH = Path.home() / "Downloads" / "Dataset.xlsx"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset.xlsx was not found here:\n{DATA_PATH}\n\n"
        "Make sure the file is named exactly Dataset.xlsx and is inside Downloads."
    )

df = pd.read_excel(DATA_PATH, engine="openpyxl")
df.columns = df.columns.astype(str).str.strip()

print("=" * 70)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 70)
print("File path:", DATA_PATH)
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())


target_column = "Machine failure"

if target_column not in df.columns:
    possible_targets = [
        col for col in df.columns
        if col.lower().replace("_", " ").strip() == "machine failure"
    ]

    if possible_targets:
        target_column = possible_targets[0]
    else:
        raise ValueError(
            "Target column was not found.\n"
            "Expected target column: Machine failure\n\n"
            f"Available columns are:\n{df.columns.tolist()}"
        )

columns_to_drop = [target_column]

if "UDI" in df.columns:
    columns_to_drop.append("UDI")

X = df.drop(columns=columns_to_drop)
y = df[target_column]

X = X.select_dtypes(include=["number"])

if X.empty:
    raise ValueError("No numeric feature columns were found.")

data = pd.concat([X, y], axis=1).dropna()

X = data[X.columns]
y = data[target_column].astype(int)

print("\nTarget column used:", target_column)

print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

numeric_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ]
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),

    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(50, 25),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )
}

results = {}

for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_test, y_pred, zero_division=0)

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    results[model_name] = {
        "trained_model": pipeline,
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "roc_thresholds": thresholds,
        "auc": auc
    }

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nConfusion Matrix Values:")
    print(f"True Negative  (TN): {tn}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")
    print(f"True Positive  (TP): {tp}")

    print("\nEvaluation Metrics:")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"AUC:         {auc:.4f}")


metrics_df = pd.DataFrame({
    model_name: {
        "Accuracy": data["accuracy"],
        "Precision": data["precision"],
        "Recall": data["recall"],
        "Specificity": data["specificity"],
        "F1-Score": data["f1_score"],
        "AUC": data["auc"]
    }
    for model_name, data in results.items()
}).T

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON TABLE")
print("=" * 70)
print(metrics_df.round(4))


for model_name, data in results.items():

    cm = data["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    image = ax.imshow(cm)

    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Normal", "Failure"])
    ax.set_yticklabels(["Normal", "Failure"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                color="black"
            )

    plt.colorbar(image)
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(8, 6))
plt.gcf().patch.set_facecolor("white")

for model_name, data in results.items():
    plt.plot(
        data["roc_fpr"],
        data["roc_tpr"],
        label=f"{model_name} AUC = {data['auc']:.3f}"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curves Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


for metric in ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "AUC"]:

    plt.figure(figsize=(8, 5))
    plt.gcf().patch.set_facecolor("white")

    values = metrics_df[metric]

    plt.bar(metrics_df.index, values)

    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)

    for i, value in enumerate(values):
        plt.text(
            i,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()


best_accuracy_model = metrics_df["Accuracy"].idxmax()
best_precision_model = metrics_df["Precision"].idxmax()
best_recall_model = metrics_df["Recall"].idxmax()
best_specificity_model = metrics_df["Specificity"].idxmax()
best_f1_model = metrics_df["F1-Score"].idxmax()
best_auc_model = metrics_df["AUC"].idxmax()

print("\n" + "=" * 70)
print("BEST MODELS BY METRIC")
print("=" * 70)

print(f"Best Accuracy:    {best_accuracy_model}")
print(f"Best Precision:   {best_precision_model}")
print(f"Best Recall:      {best_recall_model}")
print(f"Best Specificity: {best_specificity_model}")
print(f"Best F1-Score:    {best_f1_model}")
print(f"Best AUC:         {best_auc_model}")

print("\nDone.")
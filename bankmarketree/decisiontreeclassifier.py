import os
import warnings

# pyrefly: ignore [missing-import]
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "bank-full.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """Load the bank marketing CSV (semicolon-separated, quoted strings)."""
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip().strip('"') for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.strip('"')
    return df


def build_pipeline(categorical_cols, numeric_cols, **tree_kwargs) -> Pipeline:
    """Build a preprocessing + DecisionTreeClassifier pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",
    )
    clf = DecisionTreeClassifier(random_state=RANDOM_STATE, **tree_kwargs)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"  {df.shape[0]:,} rows, {df.shape[1]} columns")

    target_col = "y"
    X = df.drop(columns=[target_col])
    y = (df[target_col] == "yes").astype(int)  # 1 = subscribed, 0 = did not

    print(f"  Class balance -> yes: {y.mean():.1%} | no: {1 - y.mean():.1%}")

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
    print(f"  Categorical features: {categorical_cols}")
    print(f"  Numeric features: {numeric_cols}")

    # Train / test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Hyperparameter tuning via cross-validated grid search 
    print("\nTuning hyperparameters (5-fold CV, scoring=f1)...")
    pipeline = build_pipeline(categorical_cols, numeric_cols)
    param_grid = {
        "model__max_depth": [4, 6, 8, 10, 12, None],
        "model__min_samples_leaf": [1, 5, 10, 20],
        "model__criterion": ["gini", "entropy"],
    }
    grid = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1-score: {grid.best_score_:.4f}")

    # Evaluate on held-out test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["no", "yes"])
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Test set performance ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print(report)

    with open(os.path.join(OUTPUT_DIR, "metrics_report.txt"), "w") as f:
        f.write("Decision Tree Classifier — Bank Marketing Dataset\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Best hyperparameters: {grid.best_params_}\n")
        f.write(f"Best CV F1-score (5-fold): {grid.best_score_:.4f}\n\n")
        f.write("Test set performance\n")
        f.write("-" * 25 + "\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")
        f.write(f"ROC AUC  : {auc:.4f}\n\n")
        f.write("Classification report\n")
        f.write("-" * 25 + "\n")
        f.write(report)
        f.write("\nConfusion matrix (rows=actual, cols=predicted)\n")
        f.write(str(cm))

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["no", "yes"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature importance plot
    ohe_cols = (
        best_model.named_steps["preprocess"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_cols)
    )
    all_feature_names = list(ohe_cols) + numeric_cols
    importances = best_model.named_steps["model"].feature_importances_
    imp_df = (
        pd.DataFrame({"feature": all_feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", color="#4C72B0")
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    # Decision tree visualization (top levels only, for readability)
    plt.figure(figsize=(22, 10))
    plot_tree(
        best_model.named_steps["model"],
        max_depth=3,
        feature_names=all_feature_names,
        class_names=["no", "yes"],
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title("Decision Tree (top 3 levels)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "decision_tree_plot.png"), dpi=150)
    plt.close()

    # Save the trained pipeline (preprocessing + model)
    model_path = os.path.join(OUTPUT_DIR, "decision_tree_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\nSaved trained model to: {model_path}")
    print(f"Saved plots and report to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

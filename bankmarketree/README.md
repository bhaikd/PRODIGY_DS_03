# Decision Tree Classifier — Bank Marketing Dataset

Predicts whether a bank customer will subscribe to a term deposit (`y` = yes/no)
using demographic data (age, job, marital status, education...) and
behavioral/campaign data (call duration, number of contacts, previous
campaign outcome...).

**Dataset:** Bank Marketing dataset (`bank-full.csv`, 45,211 records, 17 columns),
from the Prodigy InfoTech Task 3 dataset repo, originally from the UCI Machine
Learning Repository:
https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%203
https://archive.ics.uci.edu/dataset/222/bank+marketing

## Folder contents

```
project/
├── decision_tree_classifier.py   # main script — run this
├── data/
│   └── bank-full.csv             # dataset (already included)
├── outputs/                      # generated after running the script
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── decision_tree_plot.png
│   ├── metrics_report.txt
│   └── decision_tree_model.joblib
└── README.md
```

How to run

1. Install dependencies:
   ```
   pip install pandas scikit-learn matplotlib seaborn joblib
   ```
2. Run the script from inside the `project/` folder:
   ```
   python decision_tree_classifier.py
   ```
3. Check the `outputs/` folder for the plots, metrics report, and saved model.

What the script does

1. **Loads** the semicolon-separated CSV and cleans quoted strings.
2. **Splits** features into categorical (one-hot encoded) and numeric
   (passed through unchanged), wrapped in an sklearn `Pipeline`.
3. **Tunes hyperparameters** (`max_depth`, `min_samples_leaf`, `criterion`)
   with 5-fold cross-validated `GridSearchCV`, optimizing F1-score (the
   dataset is imbalanced — only ~11.7% of customers subscribe).
4. **Evaluates** the best model on a held-out 20% test set: accuracy,
   precision, recall, F1, ROC AUC, confusion matrix, classification report.
5. **Visualizes** the top 3 levels of the tree and the top 15 most important
   features.
6. **Saves** the trained pipeline (preprocessing + model) as a `.joblib` file
   so it can be reloaded and used for new predictions without retraining:
   ```python
   import joblib
   model = joblib.load("outputs/decision_tree_model.joblib")
   model.predict(new_customers_df)   # new_customers_df must have the same columns as X
   ```

## Results (this run)

| Metric | Value |
|---|---|
| Accuracy | 0.90 |
| Precision (yes) | 0.58 |
| Recall (yes) | 0.52 |
| F1-score (yes) | 0.55 |
| ROC AUC | 0.85 |

The model performs well overall (90% accuracy), but — as expected for an
imbalanced dataset where only ~12% of customers subscribe — it's harder to
catch every "yes" case. Call `duration`, `poutcome` (previous campaign
outcome), and `month` are consistently the strongest predictors, which
matches the well-known finding on this dataset: how long a customer stays
on the call and whether they responded positively before are the biggest
tells for whether they'll subscribe this time.

**Note on `duration`:** the UCI documentation flags that call duration is
only known *after* the call happens, so it can't be used for a realistic
pre-call prediction model. It's kept here for classifier quality/benchmarking,
but for a deployable "who should we call" model, consider dropping it.

# Telco Customer Churn Prediction

A machine learning pipeline that predicts customer churn for a telecom provider. Also includes a feature selection and hyperparameter tuning step using genetic algorithms and particle swarm optimization.

## Overview

Telecom companies lose a lot of revenue to churn, and it's cheaper to keep an existing customer than to acquire a new one. This project trains a classifier that flags customers likely to churn based on their account, service, and billing info, so retention efforts can be targeted instead of blanket.

Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), 7,043 customers, 20 features (demographics, subscribed services, contract type, billing method, charges) plus the binary target `Churn`.

## Pipeline

### 1. Cleaning (`main.py`)

- Drops the `customerID` column, it's just an identifier.
- Fixes `TotalCharges`: 11 rows are blank because those customers are brand new (tenure = 0). These get filled with `MonthlyCharges * tenure` instead of being dropped or imputed, since the real value is just known math in this case.
- Detects binary Yes/No columns and maps them to 1/0. This checks the exact set of unique values in each column, not just what percentage matches Yes/No. Columns like `OnlineSecurity` or `MultipleLines` have a third value ("No internet service" / "No phone service") and get left alone so they're one-hot encoded properly downstream instead of turning into NaN.

### 2. Preprocessing

- Numeric features scaled with `StandardScaler`.
- Remaining categorical features one-hot encoded.
- Stratified 70/30 train/test split.

### 3. Modeling (`train_model.py`)

Trained and compared three baseline classifiers: Logistic Regression, Random Forest, and Gradient Boosting, all wrapped in an sklearn Pipeline along with preprocessing so the scaler/encoder never sees test data during fitting.

Evaluated on test AUC, 5-fold CV AUC, accuracy, and per-class precision/recall/F1. Accuracy alone isn't a great metric here since about 73% of customers don't churn.

### 4. Feature selection + tuning (`evolve_model.py`)

- Genetic algorithm (DEAP) for feature selection. Each individual is a binary mask over the encoded features, evolved through selection, crossover and mutation to maximize cross-validated AUC. Cut the feature set from 40 down to 20 columns with basically no drop in performance.
- Particle swarm optimization (pyswarms) for hyperparameter tuning of n_estimators, max_depth, and learning_rate. Particles move through the search space pulled toward their own best position and the swarm's best position, similar to how PSO was originally inspired by bird flocking behavior.

## Results

| Model | Test AUC | 5-fold CV AUC | Churn Recall | Churn F1 |
|---|---|---|---|---|
| Random Forest | 0.831 | 0.854 | 0.75 | 0.62 |
| Logistic Regression | 0.828 | 0.852 | 0.79 | 0.61 |
| Gradient Boosting | 0.825 | 0.855 | 0.51 | 0.57 |
| GA-selected features + PSO-tuned GB | 0.844 | - | - | 0.59 |

Top churn drivers from feature importance: month-to-month contracts, low tenure, fiber optic internet, high charges, and lack of online security/tech support.

The three baseline models are close on AUC, so the real difference is precision vs recall. Gradient Boosting is more precise about who it flags. Logistic Regression and Random Forest catch more actual churners but with more false alarms. Which one to use depends on whether missed churners or wasted retention outreach costs more.

## Bugs found and fixed along the way

- Silent NaN injection: the original boolean detection logic used a 70% threshold to decide if a column was Yes/No. That caught columns with a third value too, and mapped that value to NaN with no warning. Model training failed later with a NaN error that had nothing obviously to do with this. Fixed by checking the exact set of unique values per column instead of a percentage.
- Wrong array slice: `predict_proba(X_test)[:1]` was used instead of `predict_proba(X_test)[:, 1]`. The first grabs one row, the second grabs the churn-probability column for every row. This caused a `ValueError: Found input variables with inconsistent numbers of samples` error in roc_auc_score.
- Misplaced print block: the per-model print statements were outside the training loop, so only the last model's report ever printed.

## Limitations

There are customers in the dataset with identical feature profiles who churned and who didn't, so no model can get near-perfect accuracy here. The realistic ceiling is somewhere in the mid-80s% AUC. Also, because of the class imbalance (~73% no-churn), accuracy by itself isn't a meaningful headline number.

## Running it

```bash
pip install pandas numpy scikit-learn deap pyswarms matplotlib

python main.py            # clean, split, train & compare baseline models
python train_model.py     # same pipeline, saves model + plots to disk
python evolve_model.py    # GA feature selection + PSO hyperparameter tuning
```

Expects the dataset at `./data/Telco-Customer-Churn.csv`.

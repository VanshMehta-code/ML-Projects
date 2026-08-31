import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix)
RANDOM_STATE = 131

# This function reads the csv file from the file path to a data frame
def read_csv (dataset_path: str) :
    return pd.read_csv(dataset_path)

# This function cleans the dataframe
def cleaning_df (df):
    # First we make the copy of the df to the clean_df
    clean_df = df.copy ()

    # Then we remove the Customer ID column because it is not useful as all the values are unique in it
    clean_df = clean_df.drop(columns = ["customerID"])

    # Then we start with cleaning the Total Charge column which is a object column
    # So first we convert that column into numeric values
    clean_df['TotalCharges'] = pd.to_numeric(clean_df['TotalCharges'], errors='coerce')
    # Then we fill the na blocks with the total charge they are paying (Monthly charges * tenure)
    clean_df['TotalCharges'] = clean_df['TotalCharges'].fillna(clean_df['MonthlyCharges'] * clean_df['tenure'])
    return clean_df

# This function takes the dataframe and then split that data frame for training and testing purpose.
def split_data (df):
    # Here first we set up a target column, here it is "Churn" column in dataset, and also we map the yes and no to the 1 and 0
    target = df["Churn"].map({'Yes':1, 'No': 0})

    # This is where we remove the target column from the dataset so the model don't get that column while training
    X = df.drop(columns = ["Churn"])

    # So now we make a variable where we store all the categorical columns for scaling and transformation
    cat_cols = X.select_dtypes(include = "object").columns.tolist()

    # And we also seperatelt store the columns which are not objects or str
    num_cols = X.select_dtypes(exclude = "object").columns.tolist()

    # Now we do the train and test split
    # So it is a basic randomly split the dataset, here the X, target are split at 0.3 means 30% for test and 70 for train
    # Here we get 4 variable, X_train, X_test, y_train, y_test
    # X_train -> This is the 70 % of the input dataset to train
    # X_test -> This is the 30 % of the input dataset to test
    # y_train -> This is the 70 % of the target dataset to train
    # y_test -> This is the 30 % of the target dataset to test
    # Here: input dataset means the X(matrix), target dataset means target(array)
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.3, stratify=target, random_state= RANDOM_STATE
    )

    # Now we transform the column to better version for computer,
    # StandarScaler is for the numeric data and it would simply make the data on scale from a standard distribution
    # OneHotEncoder is for categorical data and it would transform it to a one numeric array form.
    preprocess = ColumnTransformer([('num', StandardScaler(), num_cols), ('cat', OneHotEncoder(handle_unknown='ignore', drop="if_binary"), cat_cols)])
    return X_train, X_test, y_train, y_test, preprocess

# This function trains and test the models for prediction
def training_testing_model (X_train, X_test, y_train, y_test, preprocess):
    # This is the results named dictionary that would be used to store the result of the models.
    results ={}

    # This is the fitted_pipe named dictionary to store the pipeline from the models we created.
    fitted_pipe= {}

    # This is the models named dictionary that carries all the different models that we are gonna train and test
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    # This loop would iterate through each model and train and test on that model.
    for name, model in models.items():
        # We first create a pipeline with the model and the preprocessed data (the data where we used the transformation code)
        pipe = Pipeline([('prep', preprocess), ('clf', model)])

        # Here we give the data the X_train and y_train data
        pipe.fit(X_train, y_train)
        # For quick recall:
        # The X_train is our dataset with 70% of random rows without targetted column
        # The y_train is our targeted column with the line of the input dataset

        # Here we now store the pipe into the dictionary
        fitted_pipe[name] = pipe

        # The proba returns probablity of 0, 1 (result) 
        # gives a matrix of (rows in X_test * total possible outcome)
        proba = pipe.predict_proba(X_test)[:,1]

        # This gives direct the class it falls in means 0 or 1 and it size is number of rows in X_test
        preds = pipe.predict(X_test)

        auc= roc_auc_score(y_test, proba)
        report= classification_report(y_test, preds, target_names=['No Churn', 'Churn'], output_dict=True)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc' ).mean()
        results[name] = {
            'test_auc': auc,
            'cv_auc_mean': cv_auc,
            'accuracy': report['accuracy'],
            'churn_precision': report['Churn']['precision'],
            'churn_recall': report['Churn']['recall'],
            'churn_f1': report['Churn']['f1-score'],
        }
        print(f"\n=== {name} ===")
        print(f"Test AUC: {auc:.4f} | 5-fold CV AUC: {cv_auc:.4f}")
        print(classification_report(y_test, preds, target_names=['No Churn', 'Churn']))
        print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    return results, fitted_pipe
def main():
    print("Hello from telco-churn-customer!")
    df = read_csv("./data/Telco-Customer-Churn.csv")
    print("[X] Reading the csv")

    clean_df = cleaning_df(df)
    print("[X] Cleaned the Dataset")

    X_train, X_test, y_train, y_test, preprocess = split_data (clean_df)
    print("[X] Spliting data completed")

    results, fitted_pipelines = training_testing_model (X_train, X_test, y_train, y_test, preprocess)
    print ("[X] Training and testing model completed")
    summary = pd.DataFrame(results).T.sort_values('test_auc', ascending=False)
    print("\n=== Model comparison ===")
    print(summary.round(4))
    best_name = summary.index[0]
    best_pipe = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name}")
if __name__ == "__main__":
    main()
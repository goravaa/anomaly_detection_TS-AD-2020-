import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier

def load_data():
    """
    Load the training and test data from CSV files.
    Adjust file paths as necessary for your environment.
    """
    train = pd.read_csv("/content/train.csv")
    test = pd.read_csv("/content/test.csv")
    return train, test

def preprocess_data(train, test):
    """
    Convert 'Date' columns to datetime, extract time features, 
    and remove unneeded columns.
    """
    # Convert the Date columns to datetime objects
    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    # Extract simple date-time features
    for df in [train, test]:
        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
        df["day"] = df["Date"].dt.day
        df["hour"] = df["Date"].dt.hour
    
    # Drop the original Date column
    train.drop(columns=["Date"], inplace=True)
    test.drop(columns=["Date"], inplace=True)
    
    return train, test

def train_and_evaluate(train_df):
    """
    Train on the complete dataset.
    Evaluate on a validation set using an 80/20 split.
    
    Returns a trained model.
    """
    # Separate features and target using the complete dataset
    X = train_df.drop(columns=["target"])
    y = train_df["target"]

    # Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Initialize the model: RandomForest with class_weight to help handle imbalance
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)

    # Validation performance
    y_pred = rf.predict(X_val)
    print("Validation Classification Report:")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("F1 Score:", f1_score(y_val, y_pred))
    
    return rf

def predict_and_submit(model, test_df, output_file="submission.csv"):
    """
    Make predictions on the test set and save them to a CSV for submission.
    """
    test_ids = test_df["ID"]
    
    # Drop the ID column if you don't want to use it as a feature
    test_features = test_df.drop(columns=["ID"], errors="ignore")

    # Make predictions
    preds = model.predict(test_features)

    # Build a DataFrame in the format "ID, target"
    submission = pd.DataFrame({"ID": test_ids, "target": preds})
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    return submission

if __name__ == "__main__":
    # 1. Load data
    train_df, test_df = load_data()
    
    # 2. Preprocess (create additional features from Date, drop Date columns)
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # 3. Train the model using the complete dataset and evaluate
    model = train_and_evaluate(train_df)
    
    # 4. Predict on test set and produce a final CSV file
    final_submission = predict_and_submit(model, test_df, output_file="submission.csv")
    print(final_submission.head())

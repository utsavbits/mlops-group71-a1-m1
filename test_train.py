import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib


# Test data loading
def test_data_loading():
    drug_df = pd.read_csv("Data/drug.csv")
    assert not drug_df.empty, "Dataframe is empty"
    assert "Drug" in drug_df.columns, "Drug column is missing"


# Test train-test split
def test_train_test_split():
    drug_df = pd.read_csv("Data/drug.csv")
    X = drug_df.drop("Drug", axis=1).values
    y = drug_df.Drug.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=125
    )
    assert len(X_train) > 0, "Training set is empty"
    assert len(X_test) > 0, "Test set is empty"
    assert len(y_train) > 0, "Training labels are empty"
    assert len(y_test) > 0, "Test labels are empty"


# Test model training
def test_model_training():
    drug_df = pd.read_csv("Data/drug.csv")
    X = drug_df.drop("Drug", axis=1).values
    y = drug_df.Drug.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=125
    )

    cat_col = [1, 2, 3]
    num_col = [0, 4]

    transform = ColumnTransformer(
        [
            ("encoder", OrdinalEncoder(), cat_col),
            ("num_imputer", SimpleImputer(strategy="median"), num_col),
            ("num_scaler", StandardScaler(), num_col),
        ]
    )
    pipe = Pipeline(
        steps=[
            ("preprocessing", transform),
            ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
        ]
    )
    pipe.fit(X_train, y_train)
    assert pipe, "Model training failed"

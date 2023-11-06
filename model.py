# Importing required libraries
import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


# Function to read and preprocess data
def read_and_preprocess():
    df = pd.read_csv()

    # Filling in missing values
    for col in df.columns:
        if df[col].isnull().sum() != 0:
            df[col].fillna(np.mean(df[col]), inplace=True)

    # Splitting into features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


# Reading in data
filepaths = [
    "C:/Users/Henry/Crime Hotspot Identification System/CRIME-INPUT/2019.csv",
    "C:/Users/Henry/Crime Hotspot Identification System/CRIME-INPUT/2020.csv",
    "C:/Users/Henry/Crime Hotspot Identification System/CRIME-INPUT/2021.csv",
    "C:/Users/Henry/Crime Hotspot Identification System/CRIME-INPUT/2022.csv",
]
data = [read_and_preprocess(fp) for fp in filepaths]
X = pd.concat([d[0] for d in data])
y = pd.concat([d[1] for d in data])

# Encoding categorical features
encoder = ce.OrdinalEncoder()
X = encoder.fit_transform(X)

# Scaling the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Performing a train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Choosing a model
rf = RandomForestRegressor()

# Defining a parameter grid
param_grid = {
    "max_depth": [5, 10, 25, 100],
    "max_features": ["sqrt", "log2", None],
    "max_samples": [100, 500, None],
}

# Creating a grid search
grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=4, n_jobs=-1)

# Training the grid search
grid.fit(X_train, y_train)

# Creating the final model
params = grid.best_params_
final_rf = RandomForestRegressor(
    max_depth=params["max_depth"],
    max_features=params["max_features"],
    max_samples=params["max_samples"],
)

# Training the final model
final_rf.fit(X_train, y_train)

# Getting mse for the testing data
preds = final_rf.predict(X_test)
final_rf.score(X_test, y_test)

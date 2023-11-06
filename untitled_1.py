# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from arcgis.gis import GIS
from arcgis.features import FeatureLayerCollection


# Function to read and preprocess data
def read_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # Filling in missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and df[col].isnull().sum() != 0:
            df[col].fillna(np.mean(df[col]), inplace=True)

    # Splitting into features and target
    X = df.select_dtypes(include=[np.number])  # Select only numeric columns
    y = df.iloc[:, -1]

    # Fill in NaN values in the target variable
    if y.isnull().sum() != 0:
        y.fillna(np.mean(y), inplace=True)

    return X, y



# Reading in data
filepaths = ["C:/Users/Henry/Crime Hotspot Identification System/CRIME-INPUT/2019.csv",
             "C:/Users/Henry/Crime Hotspot Identification System/CRIME-INPUT/2020.csv",
             "C:/Users/Henry/Crime Hotspot Identification System/CRIME-INPUT/2021.csv",
             "C:/Users/Henry/Crime Hotspot Identification System/CRIME-INPUT/2022.csv"]
data = [read_and_preprocess(fp) for fp in filepaths]
X = pd.concat([d[0] for d in data])
y = pd.concat([d[1] for d in data])

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

# Connect to ArcGIS Online
gis = GIS(api_key="AAPK455075b83f2f4592ae458f294740436aN5B2hqK2-g8bzT4Hf4nwN3WVXB5m47uShoJSqGr_4pPmmmaAajw-3hwEES8BWUek")

# Create a DataFrame with your predictions and coordinates
df_preds = pd.DataFrame({
    'Latitude': X_test['Latitude'],
    'Longitude': X_test['Longitude'],
    'Prediction': preds
})

# Convert DataFrame to spatially enabled DataFrame
sdf = pd.DataFrame.spatial.from_xy(df=df_preds, x_column='Longitude', y_column='Latitude')

# Save spatially enabled DataFrame as a feature layer
sdf.spatial.to_featurelayer(title='Crime Hotspot Predictions', gis=gis)

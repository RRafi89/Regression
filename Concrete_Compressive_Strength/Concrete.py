"""
Model: Random Forest Regressor
Model Created: Md. Rafiquzzaman Rafi
Date: 26 August, 2024
============= Data Source =============
Original Owner and Donor
Prof. I-Cheng Yeh
Department of Information Management 
Chung-Hua University, 
Hsin Chu, Taiwan 30067, R.O.C.
e-mail:icyeh@chu.edu.tw
TEL:886-3-5186511
Date Donated: August 3, 2007
=======================================
"""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

# Load and preprocess data
data_path = "Concrete_Data.csv"
data = pd.read_csv(data_path)

data = data.rename(
    columns={
        "Cement (component 1)(kg in a m^3 mixture)": "cement",
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "blast_furnace_slag",
        "Fly Ash (component 3)(kg in a m^3 mixture)": "fly_ash",
        "Water  (component 4)(kg in a m^3 mixture)": "water",
        "Superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer",
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "coarse_aggregate",
        "Fine Aggregate (component 7)(kg in a m^3 mixture)": "fine_aggregate",
        "Age (day)": "age",
        "Concrete compressive strength(MPa, megapascals) ": "compressive_strength",
    }
)

# Feature engineer the data
data['cement_coarse'] = data.cement / data.coarse_aggregate
data['cement_fine'] = data.cement / data.fine_aggregate

X = data.drop("compressive_strength", axis=1)
y = data["compressive_strength"]

# Define preprocessing for numerical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Define the full preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('transformer', PowerTransformer(method='yeo-johnson', standardize=False)),
            ('scaler', StandardScaler()),
        ]), numerical_features)
    ]
)

# Define the RandomForestRegressor model
model = RandomForestRegressor(
    max_depth=None,
    max_features="log2",
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=300,
    random_state=42,
)

# Create the full pipeline with preprocessing and model
pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Perform cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
print("Cross-Validation Scores:", cv_scores)
print("Mean R² Score on Train Data:", cv_scores.mean())

# Evaluate on Test Data
test_score = pipeline.score(X_test, y_test)
print("R² Score on Test Data:", test_score)

y_preds = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_preds)
mse = mean_squared_error(y_test, y_preds)
rmse = root_mean_squared_error(y_test, y_preds)

print(f'MAE score: {mae}\nMSE score: {mse}\nRMSE score: {rmse}')

# ====================================================================
# Save the model
# ====================================================================
# joblib.dump(pipeline, "concrete_random_forest_model.pkl")
# ====================================================================



# ====================================================================
# Load the model (when needed)
# ====================================================================
# import joblib
# import pandas as pd
# from sklearn.metrics import r2_score
# loaded_model = joblib.load('concrete_random_forest_model.pkl')

# X = pd.read_csv('X_concrete_test_data.csv')
# y = pd.read_csv('y_concrete_test_data.csv')

# y_preds = loaded_model.predict(X)
# score = r2_score(y_test, preds)
# print(score)
# ====================================================================
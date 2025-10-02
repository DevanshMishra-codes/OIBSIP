# Car Price Prediction Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load, Rename, Engineer Features
df = pd.read_csv("car data.csv") 
df.rename(columns={'Selling_Price': 'Target_Price'}, inplace=True)

# Feature Engineering
df['Car_Age'] = 2025 - df['Year']
df['Brand'] = df['Car_Name'].apply(lambda x: x.split(' ')[0])
print("Data columns:", df.columns.tolist())
print(f"Number of unique brands: {df['Brand'].nunique()}")

# Preprocessing: Feature Selection and Encoding
selected_features = [
    'Present_Price', 'Driven_kms', 'Car_Age', 
    'Fuel_Type', 'Selling_type', 'Transmission', 'Brand', 'Owner'
]

target = 'Target_Price'
df_model = df[selected_features + [target]].copy()

brand_counts = df_model['Brand'].value_counts()
infrequent_brands = brand_counts[brand_counts < 10].index 
df_model['Brand'] = df_model['Brand'].replace(infrequent_brands, 'Other')

categorical_cols = df_model.select_dtypes(include="object").columns
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

X = df_model.drop(target, axis=1)
y = df_model[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and Evaluate Multiple Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10),      
    "Lasso Regression": Lasso(alpha=0.1),    
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)          
}

results = {}

print("\n--- Model Training & Evaluation ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'RMSE': rmse, 'R2': r2}
    print(f"{name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")

# Model Comparison and Best Model Interpretation
results_df = pd.DataFrame(results).T.sort_values(by="R2", ascending=False)
print("\nModel Performance Comparison (Sorted by R²):\n", results_df)

plt.figure(figsize=(10,6))
sns.barplot(x=results_df.index, y=results_df["R2"], palette="viridis")
plt.xticks(rotation=30)
plt.ylabel("R² Score")
plt.title("Model Comparison by R² Score")
plt.show()

# Extract Feature Importance from the best tree-based model
best_tree_model_name = results_df.index[0]
if best_tree_model_name in ["Random Forest", "XGBoost"]:
    best_model = models[best_tree_model_name]
    importances = best_model.feature_importances_
    features = X.columns
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False).head(10) # Show top 10

    plt.figure(figsize=(10,5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="coolwarm")
    plt.title(f"Top 10 Feature Importance ({best_tree_model_name})")
    plt.show()

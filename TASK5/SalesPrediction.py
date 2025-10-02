import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and Explore the Data
try:
    df = pd.read_csv("Advertising.csv", index_col=0)
except FileNotFoundError:
    print("Error: Advertising.csv not found. Please ensure the file is in the correct directory.")
    exit()

print("--- Data Head ---")
print(df.head())
print("\n--- Data Info ---")
df.info() 

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f") 
plt.title("Correlation Heatmap") 
plt.show()

# Visualize relationships between each advertising channel and sales
sns.pairplot(df, x_vars=["TV", "Radio", "Newspaper"], y_vars="Sales", height=4, kind="reg")
plt.suptitle("Advertising Spend vs. Sales", y=1.02)
plt.show()

# Data Preprocessing
X = df.drop("Sales", axis=1)
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline Model: Linear Regression
lin_reg = LinearRegression() 
lin_reg.fit(X_train_scaled, y_train) 
y_pred_lr = lin_reg.predict(X_test_scaled) 

print("\n--- Baseline Linear Regression Results ---")
print(f"R² Score: {r2_score(y_test, y_pred_lr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.4f}") 

# Advanced Model: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42) 
rf_model.fit(X_train_scaled, y_train) 
y_pred_rf = rf_model.predict(X_test_scaled)

print("\n--- Random Forest (Default) Results ---")
print(f"R² Score: {r2_score(y_test, y_pred_rf):.4f}") 
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}") 

# Hyperparameter Tuning (Optimization)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20], 
    'min_samples_split': [2, 5], 
    'max_features': ['sqrt', 'log2']
}

print("\n--- Tuning Random Forest with GridSearchCV ---")
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='r2',
                           verbose=1,
                           n_jobs=-1)

grid_search.fit(X_train_scaled, y_train)
print("\nBest Parameters Found:", grid_search.best_params_) 
best_rf_model = grid_search.best_estimator_

# Evaluate the FINAL, tuned model on the unseen test data
y_pred_final = best_rf_model.predict(X_test_scaled)

print("\n--- Tuned Random Forest (Final Model) Test Results ---")
print(f"Final R² Score: {r2_score(y_test, y_pred_final):.4f}")
print(f"Final RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_final)):.4f}")
print(f"Final MAE: {mean_absolute_error(y_test, y_pred_final):.4f}")

# Feature importance
importances = best_rf_model.feature_importances_ 
feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis') 
plt.title('Feature Importance from Tuned Random Forest') 
plt.show()

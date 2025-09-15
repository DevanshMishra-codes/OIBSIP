# Iris Flower Classification using Multiple ML Models

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Iris.csv")
print("First 5 rows of dataset:\n", df.head())
print("\nDataset Info:\n")
print(df.info())

# Drop 'Id' column
df.drop(columns=["Id"], inplace=True)

# EDA
print("\nChecking for missing values:\n", df.isnull().sum())
print("\nClass Distribution:\n", df['Species'].value_counts())

# Visualize data distribution
sns.pairplot(df, hue="Species", diag_kind="kde")
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

X = df.drop("Species", axis=1).values
y = df["Species"].values

# Encode labels (Setosa, Versicolor, Virginica → 0,1,2)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Model Comparison
print("\nModel Accuracies:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

# Barplot of accuracies
plt.figure(figsize=(7,4))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.xticks(rotation=30)
plt.show()

# Cross Validation
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
cv_scores = cross_val_score(best_model, X, y, cv=5)

print(f"\nBest Model: {best_model_name}")
print("Cross-validation scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Preprocessing
ps = PorterStemmer()
stop_words = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [ps.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(text)

df['clean_message'] = df['message'].apply(clean_text)

# Features & Labels
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_message']).toarray()
y = df['label'].map({'ham':0, 'spam':1}).values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Model (Naive Bayes)
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ROC Curve
RocCurveDisplay.from_estimator(nb, X_test, y_test)
plt.title("ROC Curve - Naive Bayes")
plt.show()

def predict_message(msg):
    clean_msg = clean_text(msg)                        
    vectorized = tfidf.transform([clean_msg]).toarray() 
    prediction = nb.predict(vectorized)[0]              
    return "Spam" if prediction == 1 else "Ham"
while True:
    user_input = input("Enter a message (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    print("Prediction:", predict_message(user_input))

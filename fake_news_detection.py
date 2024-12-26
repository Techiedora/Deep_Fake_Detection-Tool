#!/usr/bin/env python3

import re  # Ensure re module is imported
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tkinter as tk
from tkinter import messagebox
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('stopwords')

# Load dataset
data = pd.read_csv('news.csv')  # Ensure your dataset is named 'news.csv'

# Data preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

data['text'] = data['text'].apply(preprocess_text)
X = data['text']
y = data['label']  # Assuming 'label' column contains 'FAKE' or 'REAL'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TfidfVectorizer with n-grams and max features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=20000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize RandomForestClassifier with hyperparameter tuning using GridSearchCV
classifier = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(classifier, param_grid, scoring='accuracy', cv=2)  # Reduced cv to 2
grid_search.fit(X_train_tfidf, y_train)
classifier = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict news article
def predict_news():
    article = text_entry.get("1.0", tk.END).strip()
    if article:
        article = preprocess_text(article)
        article_tfidf = tfidf_vectorizer.transform([article])
        prediction = classifier.predict(article_tfidf)
        messagebox.showinfo("Prediction Result", f"The news article is: {prediction[0]}")
    else:
        messagebox.showwarning("Input Error", "Please enter a news article.")

# Create the main window
root = tk.Tk()
root.title("Fake News Detection")

# Create a label and text entry for the news article input
label = tk.Label(root, text="Enter News Article:")
label.pack(pady=10)

text_entry = tk.Text(root, height=10, width=50)
text_entry.pack(pady=10)

# Create a button to make predictions
predict_button = tk.Button(root, text="Predict", command=predict_news)
predict_button.pack(pady=20)

# Display accuracy in GUI
accuracy_label = tk.Label(root, text=f"Model Accuracy: {accuracy:.2f}", font=("Helvetica", 12))
accuracy_label.pack(pady=10)

# Run the application
root.mainloop()

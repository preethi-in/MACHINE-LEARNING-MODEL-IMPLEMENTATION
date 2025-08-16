pip install pandas scikit-learn
# Spam Email Detection using Scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 2: Encode labels (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Step 4: Convert text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test_tfidf)

# Step 7: Evaluate model
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Test custom input
sample = ["Congratulations! You've won a free ticket to Bahamas. Call now!"]
sample_tfidf = vectorizer.transform(sample)
print("\nCustom Prediction:", "Spam" if model.predict(sample_tfidf)[0] == 1 else "Ham")

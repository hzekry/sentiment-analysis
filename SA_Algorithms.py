
import joblib
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
data = pd.read_json('pet_related_reviews.json')
data = data.head(1000)
# Data preprocessing
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [w for w in words if not w in set(stopwords.words('english'))]
    text = ' '.join(words)
    return text

data['cleaned_text'] = data['text'].apply(preprocess_text)

# Sentiment analysis
sia = SentimentIntensityAnalyzer()
data['sentiment_score'] = data['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Define sentiment labels based on sentiment scores and stars
data['sentiment_label'] = np.where(
    ((data['sentiment_score'] > 0.6) & (data['stars'] > 3)),
    'positive',
    np.where(
        ((data['sentiment_score'] >= -0.5) & (data['sentiment_score'] <= 0.5)) & (data['stars'] == 3),
        'neutral',
        'negative'
    )
)


# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['sentiment_label'], test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize the models
svm_model = SVC(kernel='linear')
nb_model = MultinomialNB()
rf_model = RandomForestClassifier(n_estimators=100)
gb_model = GradientBoostingClassifier(n_estimators=100)

# Fit the models
svm_model.fit(X_train_vectorized, y_train)
nb_model.fit(X_train_vectorized, y_train)
rf_model.fit(X_train_vectorized, y_train)
gb_model.fit(X_train_vectorized, y_train)

# Perform predictions
svm_pred = svm_model.predict(X_test_vectorized)
nb_pred = nb_model.predict(X_test_vectorized)
rf_pred = rf_model.predict(X_test_vectorized)
gb_pred = gb_model.predict(X_test_vectorized)

# Evaluate the models
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average='macro')
svm_recall = recall_score(y_test, svm_pred, average='macro')
svm_precision = precision_score(y_test, svm_pred, average='macro', zero_division=1)

nb_accuracy = accuracy_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred, average='macro')
nb_recall = recall_score(y_test, nb_pred, average='macro')
nb_precision = precision_score(y_test, nb_pred, average='macro', zero_division=1)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='macro')
rf_recall = recall_score(y_test, rf_pred, average='macro')
rf_precision = precision_score(y_test, rf_pred, average='macro', zero_division=1)

gb_accuracy = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred, average='macro')
gb_recall = recall_score(y_test, gb_pred, average='macro')
gb_precision = precision_score(y_test, gb_pred, average='macro', zero_division=1)

print('SVM Accuracy:', svm_accuracy)
print('SVM F1-score:', svm_f1)
print('SVM Recall:', svm_recall)
print('SVM Precision:', svm_precision)
print('Naive Bayes Accuracy:', nb_accuracy)
print('Naive Bayes F1-score:', nb_f1)
print('Naive Bayes Recall:', nb_recall)
print('Naive Bayes Precision:', nb_precision)
print('Random Forest Accuracy:', rf_accuracy)
print('Random Forest F1-score:', rf_f1)
print('Random Forest Recall:', rf_recall)
print('Random Forest Precision:', rf_precision)
print('Gradient Boosting Accuracy:', gb_accuracy)
print('Gradient Boosting F1-score:', gb_f1)
print('Gradient Boosting Recall:', gb_recall)
print('Gradient Boosting Precision:', gb_precision)
# Calculate combined score for SVM
svm_combined_score = (svm_accuracy + svm_f1 + svm_recall + svm_precision) / 4

# Combined score for Naive Bayes
nb_combined_score = (nb_accuracy + nb_f1 + nb_recall + nb_precision) / 4

#Combined score for Random Forest
rf_combined_score = (rf_accuracy + rf_f1 + rf_recall + rf_precision) / 4

#Combined score for Gradient Boosting
gb_combined_score = (gb_accuracy + gb_f1 + gb_recall + gb_precision) / 4
# Choose the best model
best_combined_score = max(svm_combined_score, nb_combined_score, rf_combined_score, gb_combined_score)

if best_combined_score == svm_combined_score:
    best_algorithm = 'SVM'
    best_model = svm_model
elif best_combined_score == nb_combined_score:
    best_algorithm = 'Naive Bayes'
    best_model = nb_model
elif best_combined_score == rf_combined_score:
    best_algorithm = 'Random Forest'
    best_model = rf_model
else:
    best_algorithm = 'Gradient Boosting'
    best_model = gb_model

print('Best Algorithm:', best_algorithm)

# Use the best model for predictions on the entire dataset
all_predictions = best_model.predict(vectorizer.transform(data['cleaned_text']))
data['predicted_sentiment'] = all_predictions

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_sklearn_model(best_model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('sentiment_model.tflite', 'wb') as f:
    f.write(tflite_model)
# # Save the dataset with predicted sentiment labels to a new JSON file
# data.to_json('pet_related_reviews_with_sentiment.json', orient='records', date_format='iso')

# # Save the best model and vectorizer for future use
# joblib.dump(best_model, 'best_model.pkl')
# joblib.dump(vectorizer, 'vectorizer.pkl')

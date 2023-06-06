import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
import tensorflow as tf

# Load the dataset
data = pd.read_json('pet_related_reviews.json')
data = data.head(40000)

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

# Fit the models
svm_model.fit(X_train_vectorized, y_train)

# Perform predictions
svm_pred = svm_model.predict(X_test_vectorized)

# Evaluate the models
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average='macro')
svm_recall = recall_score(y_test, svm_pred, average='macro')
svm_precision = precision_score(y_test, svm_pred, average='macro', zero_division=1)

print('SVM Accuracy:', svm_accuracy)
print('SVM F1-score:', svm_f1)
print('SVM Recall:', svm_recall)
print('SVM Precision:', svm_precision)

# Use the best model for predictions on the entire dataset
all_predictions = svm_model.predict(vectorizer.transform(data['cleaned_text']))
data['predicted_sentiment'] = all_predictions

# Create a Keras model
num_features = X_train_vectorized.shape[1]
num_classes = 3

keras_model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=num_classes, activation='softmax', input_shape=(num_features,))
])

# Reshape SVM weights to match the Dense layer
svm_weights = svm_model.coef_.T.toarray()
svm_intercept = svm_model.intercept_

svm_weights_reshaped = np.zeros((num_features, num_classes))
svm_weights_reshaped[:, 0] = svm_weights[:, 0]  # positive class
svm_weights_reshaped[:, 2] = svm_weights[:, 1]  # negative class

# Set the weights and biases in the Keras model
keras_model.layers[0].set_weights([svm_weights_reshaped, svm_intercept])

# Compile the model
keras_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Convert sentiment labels to one-hot encoded vectors
label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
y_train_encoded = tf.keras.utils.to_categorical(y_train.map(label_mapping), num_classes)
y_test_encoded = tf.keras.utils.to_categorical(y_test.map(label_mapping), num_classes)

# Perform predictions using the Keras model
keras_pred = keras_model.predict(X_test_vectorized)
keras_pred_classes = np.argmax(keras_pred, axis=1)
keras_pred_labels = pd.Series(keras_pred_classes).map({0: 'positive', 1: 'neutral', 2: 'negative'})

# Evaluate the Keras model
keras_accuracy = accuracy_score(y_test, keras_pred_labels)
keras_f1 = f1_score(y_test, keras_pred_labels, average='macro')
keras_recall = recall_score(y_test, keras_pred_labels, average='macro')
keras_precision = precision_score(y_test, keras_pred_labels, average='macro', zero_division=1)

print('Keras Accuracy:', keras_accuracy)
print('Keras F1-score:', keras_f1)
print('Keras Recall:', keras_recall)
print('Keras Precision:', keras_precision)
 
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('sentiment_model.tflite', 'wb') as f:
    f.write(tflite_model)
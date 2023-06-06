# import pandas as pd
# import numpy as np
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.svm import SVC
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score

# # Load the airline review dataset
# data = pd.read_json('pet_related_reviews.json')
# data = data.head(1000)

# # Preprocessing functions
# def preprocess_text(text):
#     # Tokenize, remove stop words, URLs, and digits
#     words = word_tokenize(text)
#     words = [w for w in words if not w in set(stopwords.words('english'))]
#     words = [w for w in words if not re.match(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', w)]
#     words = [w for w in words if not w.isdigit()]
    
#     # Remove punctuation and emoticons
#     words = [w for w in words if w.isalpha()]
    
#     return ' '.join(words)

# # Apply preprocessing to the review text
# data['cleaned_text'] = data['text'].apply(preprocess_text)

# # Extract features using TF-IDF and Bag of Words
# vectorizer_ti = TfidfVectorizer()
# vectorizer_bow = CountVectorizer()
# X_ti = vectorizer_ti.fit_transform(data['cleaned_text'])
# X_bow = vectorizer_bow.fit_transform(data['cleaned_text'])

# # Define the sentiment labels (positive or negative) based on 'stars' column
# y = np.where(data['stars'] > 3, 1, 0).reshape(-1)


# # Initialize scores
# ml_scores_ti = np.zeros((X_ti.shape[0], X_ti.shape[0]))
# ml_scores_bow = np.zeros((X_bow.shape[0], X_bow.shape[0]))

# # Apply ML algorithms and calculate scores
# for i in range(X_ti.shape[0]):
#     # SVM with TF-IDF features
#     svm_model_ti = SVC(kernel='linear')
#     svm_model_ti.fit(X_ti, y)
#     ml_scores_ti[i] = svm_model_ti.score(X_ti, y)

#     # Naive Bayes with Bag of Words features
#     nb_model_bow = MultinomialNB()
#     nb_model_bow.fit(X_bow, y)
#     ml_scores_bow[i] = nb_model_bow.score(X_bow, y)

# # Compare and select the best results
# svm_avg_score_ti = np.mean(ml_scores_ti)
# nb_avg_score_bow = np.mean(ml_scores_bow)

# if svm_avg_score_ti > nb_avg_score_bow:
#     best_algorithm = 'SVM with TF-IDF'
#     best_scores = ml_scores_ti
#     best_model = svm_model_ti
# else:
#     best_algorithm = 'Naive Bayes with Bag of Words'
#     best_scores = ml_scores_bow
#     best_model = nb_model_bow

# # Print the best algorithm and its scores
# print('Best Algorithm:', best_algorithm)
# print('Scores:')
# print(best_scores)


# print('Best Algorithm:', best_algorithm)
# accuracy = accuracy_score(y, best_model.predict(X_ti if best_algorithm == 'SVM with TF-IDF' else X_bow))
# print('Accuracy:', accuracy)

# # Add scores column to the original data
# data['scores'] = best_scores.tolist()

# # Save the data to a new JSON file
# data.to_json('pet_related_reviews_with_scores.json', orient='records')
# # End of the code
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

# Load the review dataset
data = pd.read_json('pet_related_reviews.json')
data = data.head(1000)

# Preprocessing functions
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [w for w in words if not w in set(stopwords.words('english'))]
    words = [w for w in words if not re.match(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', w)]
    words = [w for w in words if not w.isdigit()]
    words = [w for w in words if w.isalpha()]
    return ' '.join(words)

data['cleaned_text'] = data['text'].apply(preprocess_text)

# Drop rows with empty 'cleaned_text' values
data = data.dropna(subset=['cleaned_text'])

# Split the dataset into training and testing sets
X = data['cleaned_text']
y = np.where(data['stars'] > 3, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply CountVectorizer on the training and testing data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict on the entire dataset
X = vectorizer.transform(data['cleaned_text'])
svm_pred = svm_model.predict(X)
nb_pred = nb_model.predict(X)

# Calculate F1-scores
svm_f1 = f1_score(y, svm_pred)
nb_f1 = f1_score(y, nb_pred)

print(svm_f1)
print(nb_f1)
# Choose the best model based on F1-score
if svm_f1 > nb_f1:
    best_algorithm = 'SVM'
    best_model = svm_model
else:
    best_algorithm = 'Naive Bayes'
    best_model = nb_model

# Add sentiment scores as a new column
X = vectorizer.transform(data['cleaned_text'])
data['sentiment_score'] = best_model.predict(X)

# Save the data to a new JSON file
data.to_json('pet_related_reviews_with_scores.json', orient='records')

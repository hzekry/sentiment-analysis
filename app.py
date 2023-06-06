from flask import Flask, request, jsonify
import re
import nltk
import numpy
import pandas
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import joblib

app = Flask(__name__)

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [w for w in words if not w in set(stopwords.words('english'))]
    text = ' '.join(words)
    return text

@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
    data = request.get_json()
    review = data['review']
    
    # Load the saved model and vectorizer
    best_model = joblib.load('best_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    cleaned_review = preprocess_text(review)
    review_vectorized = vectorizer.transform([cleaned_review])
    sentiment_label = best_model.predict(review_vectorized)[0]
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(cleaned_review)['compound']

    response = {
        'sentiment_score': sentiment_score,
        'sentiment_label': sentiment_label
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()

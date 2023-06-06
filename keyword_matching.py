import json
from collections import Counter
import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate('petservices-44962-firebase-adminsdk-4kfw7-ad357c5c0d.json')
firebase_admin.initialize_app(cred)

# Get a Firestore client
db = firestore.client()

# Read the reviews from the JSON file
with open('pet_related_reviews_with_sentiment.json', 'r') as f:
    reviews_data = json.load(f)

# Convert the reviews data to a DataFrame
data = pd.DataFrame(reviews_data)

# Group the reviews by business
grouped_data = data.groupby('business_id')

# Iterate over each business group
# Iterate over each business group
# Iterate over each business group
for business_id, group in grouped_data:
    # Check if the business document exists in Firestore
    doc_ref = db.collection('businesses').document(business_id)
    if not doc_ref.get().exists:
        print(f"Skipping business {business_id}. Document not found in Firestore.")
        continue
    
    # Concatenate the cleaned_text for each business into a single text
    reviews_text = ' '.join(group['cleaned_text'])
    
    # Extract the most common words from the cleaned_text
    word_counts = Counter(reviews_text.split())
    most_common_words = word_counts.most_common(10)  # Change the number as per your requirement
    
    # Create a map of words with sentiment and count as values
    common_words_sentiment = {}
    for word, count in most_common_words:
        # Get the sentiment labels for the reviews containing the word
        word_reviews = group[group['cleaned_text'].str.contains(word)]
        word_sentiments = word_reviews['sentiment_label']
        
        # Calculate the majority sentiment based on the sentiment labels
        positive_count = sum(word_sentiments == 'positive')
        negative_count = sum(word_sentiments == 'negative')
        
        if positive_count > negative_count:
            sentiment = 'positive'
        elif positive_count < negative_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Add word count and sentiment to the map
        common_words_sentiment[word] = {'count': count, 'sentiment': sentiment}
    
    # Update the Firestore document for the business
    doc_ref.update({
        'common_words_sentiment': common_words_sentiment
    })
    print(f"Keywords added to business {business_id}")

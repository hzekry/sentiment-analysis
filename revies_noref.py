
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# initialize Firebase app
cred = credentials.Certificate('petservices-44962-firebase-adminsdk-4kfw7-3b1315ad3d.json')
firebase_admin.initialize_app(cred)

# create reference to Firestore database
db = firestore.client()

# open the pet-related businesses JSON file
with open('pet_related_reviews.json', 'r', encoding='utf-8') as f:
    pet_related_reviews = json.load(f)

## iterate over the reviews
for review in pet_related_reviews:
    # check if a review with the same date already exists
    query = db.collection('reviews').where('date', '==', review['date']).get(timeout=700000)
    if len(query) > 0:
        # a review with the same date already exists, skip this one
        continue

    # add the new review to Firestore
    db.collection('reviews').add(review)
    
print('All reviews added successfuly')

import firebase_admin
from firebase_admin import credentials, firestore

# Initialize the Firebase Admin SDK
# initialize Firebase app
cred = credentials.Certificate('petservices-44962-firebase-adminsdk-4kfw7-3b1315ad3d.json')
firebase_admin.initialize_app(cred)

from firebase_admin import firestore

# Initialize Firestore client
db = firestore.client()

# Get all reviews documents
reviews_ref = db.collection('reviews')
reviews_docs = reviews_ref.get()

# Create a dictionary of unique identifiers and document IDs
unique_ids = {}
for review in reviews_docs:
    data = review.to_dict()
    unique_id = (data['user_id'], data['business_id'], data['review_text'], data['date_time'])
    doc_id = review.id
    if unique_id in unique_ids:
        # Delete duplicate document
        db.collection('reviews').document(doc_id).delete()
    else:
        unique_ids[unique_id] = doc_id

# Iterate through businesses collection and remove any references to deleted reviews
businesses_ref = db.collection('businesses')
businesses_docs = businesses_ref.get()

for business in businesses_docs:
    data = business.to_dict()
    if 'reviews' in data:
        updated_reviews = []
        for review_id in data['reviews']:
            if review_id in unique_ids.values():
                # Review was deleted, remove reference
                continue
            updated_reviews.append(review_id)
        # Update reviews field in Firestore
        businesses_ref.document(business.id).update({'reviews': updated_reviews})

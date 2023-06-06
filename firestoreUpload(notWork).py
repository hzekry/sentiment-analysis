import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# initialize Firebase app
cred = credentials.Certificate('petservices-44962-firebase-adminsdk-4kfw7-3b1315ad3d.json')
firebase_admin.initialize_app(cred)

# create reference to Firestore database
db = firestore.client()

# with open('yelp_academic_dataset_business.json', 'r', encoding='utf-8') as f:
#     for line in f:
#         business = json.loads(line)
#         categories = business.get('categories')
#         if categories and 'Pets' in categories:
#             # create a document for the pet-related business
#             doc_ref = db.collection('businesses').document(business['business_id'])
#             doc_ref.set({
#                 'name': business['name'],
#                 'address': business['address'],
#                 'city': business['city'],
#                 'state': business['state'],
#                 'stars': business['stars'],
#                 'is_open': business['is_open'],
#                 'hours': business['hours'],
#                 'attributes': business['attributes'],
#                 'categories': business['categories'],

              
#             })
with open('reviews_chunk1.json', 'r', encoding='utf-8') as f:
    for line in f:
        review = json.loads(line)
        review_id = review['review_id']
        # check if the review is for a pet-related business
        business_id = review['business_id']
        business_ref = db.collection('businesses').document(business_id)
        business_doc = business_ref.get()
        if business_doc.exists and 'Pets' in business_doc.to_dict()['categories']:
            review_ref = db.collection('reviews').document(review_id)
            review_doc = review_ref.get()
            if not review_doc.exists:
            # add the review to the business document
                review_data = {
                    'text': review['text'],
                    'stars': review['stars'],
                    'date' : review['date']
                
                }
                review_ref.set(review_data)
                business_ref.update({
                    'reviews': firestore.ArrayUnion([review_ref])
                })


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
with open('pet_related_businesses.json', 'r', encoding='utf-8') as f:
    pet_related_businesses = json.load(f)

# iterate over the businesses
for business in pet_related_businesses:
    # check if the business already exists in Firestore
    existing_business = db.collection('businesses').document(business['business_id']).get()
    if existing_business.exists:
        # skip the business if it already exists
        continue
    
    # add the new business to Firestore
    db.collection('businesses').document(business['business_id']).set(business)

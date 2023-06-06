
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
with open('customers.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Add each document to the customers collection
for doc in data:
    db.collection('Customer').document().set(doc)

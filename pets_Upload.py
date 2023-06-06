
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import random

# initialize Firebase app
cred = credentials.Certificate('petservices-44962-firebase-adminsdk-4kfw7-ad357c5c0d.json')
firebase_admin.initialize_app(cred)

# create reference to Firestore database
db = firestore.client()

# open the JSON file
with open('pets.json', 'r', encoding='utf-8') as f:
    pets_data = json.load(f)

# Get all customer documents from Firestore
customer_docs = db.collection('Customer').get()

# Create a list of customer IDs
customer_ids = [doc.id for doc in customer_docs]

# Loop through each pet in the pets data and assign a random customer ID
for pet in pets_data:
    # Choose a random customer ID
    customer_id = random.choice(customer_ids)
    
    # Add the customer_id field to the pet data
    pet['customer_id'] = customer_id

    pet['imageUrl'] = ' '
    
    # Add the pet to Firestore
    db.collection('pet').add(pet)

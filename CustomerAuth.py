import firebase_admin
from firebase_admin import credentials, firestore, auth
import json


# initialize Firebase app
cred = credentials.Certificate('petservices-44962-firebase-adminsdk-4kfw7-ad357c5c0d.json')
firebase_admin.initialize_app(cred)

# create reference to Firestore database
db = firestore.client()


# Read customers data from JSON file
with open("customers.json") as f:
    customers = json.load(f)

# Loop through customers and add them to Firestore
for customer in customers:
    # Generate random password
    password = '123456'
    
    # Create authentication user
    user = auth.create_user(email=customer["email"], password=password)
    
    # Add customer document to Firestore with authentication UID as document ID
    doc_ref = db.collection("Customer").document(user.uid)
    doc_ref.set({
        "name": customer["name"],
        "email": customer["email"],
        "phone_number" : customer["phone_number"],
        "country" : customer["country"],
        "state" : customer["state"],
        "image" : '',
        # Add more fields as needed
    })
    
    print(f"Customer {customer['name']} added with UID {user.uid}")

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json
import random

# initialize Firebase app
cred = credentials.Certificate('petservices-44962-firebase-adminsdk-4kfw7-ad357c5c0d.json')
firebase_admin.initialize_app(cred)

# create reference to Firestore database
db = firestore.client()

# open the pet-related businesses JSON file
with open('medicalRecords.json', 'r', encoding='utf-8') as f:
    medical_records = json.load(f)

# Choose a random pet ID and business ID with category "veterinary"
pet_ids = [doc.id for doc in db.collection('pet').stream()]
random_pet_id = random.choice(pet_ids)

businesses = db.collection('businesses').stream()
vet_businesses = [doc for doc in businesses if 'Veterinarians' in doc.to_dict().get('categories', '')]

random_business = random.choice(vet_businesses)
random_business_id = random_business.id

# Add the random pet ID and business ID to each medical record document
for record in medical_records:
    record['pet_id'] = random_pet_id
    record['vet_id'] = random_business_id

    # Add the record to Firestore
    db.collection('medicalRecord').add(record)

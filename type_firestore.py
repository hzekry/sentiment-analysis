import random
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate('petservices-44962-firebase-adminsdk-4kfw7-ad357c5c0d.json')
firebase_admin.initialize_app(cred)

# Create a Firestore client
db = firestore.client()

# Function to generate a random selection of animals
def get_random_animals():
    types = ['cat', 'dog', 'bird', 'fish', 'snake', 'guinea pig', 'rabbit']
    number_of_animals = random.randint(1, len(types))
    animals = random.sample(types, number_of_animals)
    return animals

# Update the "type" field for each document in the "businesses" collection
businesses_ref = db.collection('businesses')
businesses = businesses_ref.get()

for business in businesses:
    animals = get_random_animals()
    business_ref = businesses_ref.document(business.id)
    business_ref.update({'type': animals})
    print('Field "type" added to document: {}'.format(business_ref.id))

print('Field "type" has been added with random animals to all documents in the "businesses" collection.')

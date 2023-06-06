import firebase_admin
from firebase_admin import credentials, firestore

# Initialize the Firebase Admin SDK
# initialize Firebase app
cred = credentials.Certificate('petservices-44962-firebase-adminsdk-4kfw7-3b1315ad3d.json')
firebase_admin.initialize_app(cred)

# Get a reference to the businesses collection
businesses_ref = firestore.client().collection("businesses")

# Iterate over the documents in the businesses collection
for doc in businesses_ref.stream():

    # Get the current value of the categories field
    categories = doc.get("categories")
    
    if isinstance(categories, str):
        # Split the categories string by comma and store as an array
        categories_arr = categories.split(",")
        
        # Update the categories field with the new array value
        doc_ref = businesses_ref.document(doc.id)
        doc_ref.update({"categories": categories_arr})

print("Categories updated successfully.")

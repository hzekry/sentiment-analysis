import json
import firebase_admin
from firebase_admin import credentials, firestore, storage

# Initialize Firebase Admin SDK
cred = credentials.Certificate('petservices-44962-firebase-adminsdk-4kfw7-ad357c5c0d.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'petservices-44962.appspot.com'
})
db = firestore.client()
bucket = storage.bucket()

# Read pet_related_photos.json file
with open('pet_related_photos.json') as json_file:
    data = json.load(json_file)

# Iterate through photo_id and business_id pairs
for pair in data:
    photo_id = pair['photo_id']
    business_id = pair['business_id']

    # Upload photo to Firestore storage
    photo_path = 'pet_photos/{}.jpg'.format(photo_id)
    blob = bucket.blob(photo_path)
    blob.upload_from_filename(photo_path)
    photo_url = blob.public_url

    # Update business document with the photo URL
    business_image_path = 'business_images/{}.jpg'.format(business_id)
    business_ref = db.collection('businesses').document(business_id)
    business_ref.update({'image': business_image_path})

    # Retrieve business document from Firestore
    business_doc = business_ref.get()

    if business_doc.exists:
        # Check if the 'images' field exists
        if 'images' in business_doc.to_dict():
            # Update images array in the business document
            images = business_doc.get('images') or []
            images.append(photo_url)
            business_ref.update({'images': images})
        else:
            # Create 'images' field and assign it as an array
            business_ref.update({'images': [photo_url]})
        
        print('Added photo URL to business:', business_id)
    else:
        print('Business not found:', business_id)

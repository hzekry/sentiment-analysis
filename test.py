import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore



# initialize Firebase app
cred = credentials.Certificate('petservices-44962-firebase-adminsdk-4kfw7-ad357c5c0d.json')
firebase_admin.initialize_app(cred)

# create reference to Firestore database
db = firestore.client()

# open the reviews JSON file
with open('pet_related_reviews_with_sentiment.json', 'r', encoding='utf-8') as f:
    pet_related_reviews = json.load(f)[:60000]

# process each review in the JSON file
# for review in pet_related_reviews:
#     try:
#         # check if the review already exists in Firestore
#         existing_review = db.collection('reviews').where('business_id', '==', review['business_id']).where('text', '==', review['text']).get(timeout=60000000)
#         if existing_review:
#             # skip the review if it already exists
#             continue
existing_reviews = {}
for review in db.collection('reviews').get():
    existing_reviews[(review.get('business_id'), review.get('text'))] = True

# process each review in the JSON file
for review in pet_related_reviews:
    try:
    # check if the review already exists in the set/dictionary
        if (review['business_id'], review['text']) in existing_reviews:
        # skip the review if it already exists
            continue
    except:
        continue

    # split the review text into parts of max_length characters
    max_length = 1500  
    review_text = review['text']
    review_length = len(review_text)
    if review_length > max_length:
        num_parts = (review_length-1) // max_length + 1 # number of parts required to split the review
        for i in range(num_parts):
            # get the text for this part
            start_index = i*max_length
            end_index = min((i+1)*max_length, review_length)
            text_part = review_text[start_index:end_index]
            if i == 0:
                review['text'] = text_part
            else:
                review[f'text_{i+1}'] = text_part
    # add the review to Firestore
    review_ref = db.collection('reviews').document()
    review_ref.set(review)
    business_ref = db.collection('businesses').document(review['business_id'])
    business_ref.update({'reviews': firestore.ArrayUnion([review_ref])})

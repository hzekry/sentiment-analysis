import json

pet_related_reviews = []

# Read reviews file into memory
with open('pet_related_businesses.json', 'r', encoding='utf-8') as f:
    pet_related_businesses = json.load(f)

# Extract business IDs and store in a list
pet_related_business_ids = [business['business_id'] for business in pet_related_businesses]
with open('yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
    for line in f:
        review = json.loads(line)
        business_id = review.get('business_id')
        if business_id in pet_related_business_ids:
            review_data = {
                'text': review['text'],
                'stars': review['stars'],
                'date': review['date'],
                'business_id': review['business_id']
            }
            pet_related_reviews.append(review_data)

# Save the filtered reviews to a new JSON file
with open('pet_related_reviews.json', 'w', encoding='utf-8') as f:
    json.dump(pet_related_reviews, f)

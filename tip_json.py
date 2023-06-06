import json

pet_related_tip = []

# Read reviews file into memory
with open('pet_related_businesses.json', 'r', encoding='utf-8') as f:
    pet_related_businesses = json.load(f)

# Extract business IDs and store in a list
pet_related_business_ids = [business['business_id'] for business in pet_related_businesses]
with open('yelp_academic_dataset_tip.json', 'r', encoding='utf-8') as f:
    for line in f:
        tip = json.loads(line)
        business_id = tip.get('business_id')
        if business_id in pet_related_business_ids:
            tip_data = {
                'text': tip['text'],
                'compliment_count': tip['compliment_count'],
                'date': tip['date'],
                'business_id': tip['business_id']
            }
            pet_related_tip.append(tip_data)

# Save the filtered reviews to a new JSON file
with open('pet_related_tip.json', 'w', encoding='utf-8') as f:
    json.dump(pet_related_tip, f)


import json

pet_related_businesses = []

with open('yelp_academic_dataset_business.json', 'r', encoding='utf-8') as f:
    for line in f:
        business = json.loads(line)
        categories = business.get('categories')
        if categories and 'Pets' in categories:
            # add the pet-related business to the list
            pet_related_businesses.append(business)

# save the filtered businesses to a new JSON file
with open('pet_related_businesses.json', 'w', encoding='utf-8') as f:
    json.dump(pet_related_businesses, f)

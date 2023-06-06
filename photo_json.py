import json
import os
import shutil

# Path to the JSON files
pet_related_businesses_file = 'pet_related_businesses.json'
photos_file = 'photos.json'

# Path to the folder containing the photos
photos_folder = 'photos'

# Output file paths
output_json_file = 'pet_related_photos.json'
output_photos_folder = 'pet_photos'

# Load pet related businesses data
with open(pet_related_businesses_file) as pet_file:
    pet_data = json.load(pet_file)

# Create a dictionary of pet-related business IDs
pet_business_ids = {item['business_id'] for item in pet_data}

# Load photos data
with open(photos_file) as photos_file:
    photos_data = json.load(photos_file)

# Create a list to store the matched photos
matched_photos = []

os.makedirs(output_photos_folder, exist_ok= True)
# Iterate over the photos data
for photo in photos_data:
    business_id = photo['business_id']
    photo_id = photo['photo_id']
    if business_id in pet_business_ids:
        # Copy the photo to the output folder
        photo_path = os.path.join(photos_folder, f"{photo_id}.jpg")
        output_path = os.path.join(output_photos_folder, f"{photo_id}.jpg")
        shutil.copy(photo_path, output_path)

        # Append the matched photo information to the list
        matched_photos.append(photo)

# Save the matched photos data to a new JSON file
with open(output_json_file, 'w') as output_file:
    json.dump(matched_photos, output_file, indent=4)

import random
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate('petservices-44962-firebase-adminsdk-4kfw7-ad357c5c0d.json')  
firebase_admin.initialize_app(cred)
db = firestore.client()

# Function to generate 5 random time slots per day based on business hours
def generate_time_slots(hours):
    time_slots = {}

    if hours is None:
        return time_slots

    for day, day_hours in hours.items():
        if day_hours == '0:0-0:0':
            continue

        opening_time, closing_time = day_hours.split('-')
        opening_hour, opening_minute = map(int, opening_time.split(':'))
        closing_hour, closing_minute = map(int, closing_time.split(':'))

        if opening_hour > closing_hour or (opening_hour == closing_hour and opening_minute >= closing_minute):
            # Handle case when opening time is greater than or equal to closing time
            continue

        opening_total_minutes = opening_hour * 60 + opening_minute
        closing_total_minutes = closing_hour * 60 + closing_minute

        time_slots[day] = []
        existing_time_slots = set()  # Keep track of existing time slots

        while len(time_slots[day]) < 5:
            total_minutes = random.randint(opening_total_minutes, closing_total_minutes)
            hour = total_minutes // 60
            minute = total_minutes % 60
            time_slot = f"{hour}:{minute:02d}"

            # Check if the time slot already exists
            if time_slot in existing_time_slots:
                continue

            existing_time_slots.add(time_slot)
            time_slots[day].append(time_slot)

    return time_slots

# Add time slots to all businesses based on their hours
businesses_ref = db.collection('businesses')
businesses = businesses_ref.get()
for business in businesses:
    business_id = business.id
    business_data = business.to_dict()
    hours = business_data.get('hours')
    time_slots = generate_time_slots(hours)

    # Update the document in Firestore with the generated time slots
    business_ref = businesses_ref.document(business_id)
    business_ref.update({'timeSlots': time_slots})
    print(f"Time slots added to business {business_id}")

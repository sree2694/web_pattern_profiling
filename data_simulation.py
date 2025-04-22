import random
import pandas as pd

def generate_fake_data(num_users=100):
    domains = ['google.com', 'youtube.com', 'facebook.com', 'instagram.com', 'twitter.com',
               'wikipedia.org', 'amazon.com', 'reddit.com', 'netflix.com', 'stackoverflow.com']
    age_groups = ['18-25', '26-35', '36-50']
    genders = ['male', 'female']

    data = []
    for i in range(num_users):
        user_data = {
            'user_id': i,
            'age_group': random.choice(age_groups),
            'gender': random.choice(genders),
            'browsing_sequence': random.sample(domains, random.randint(3, 6))
        }
        data.append(user_data)
    return pd.DataFrame(data)

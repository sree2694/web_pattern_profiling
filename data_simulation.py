import random
import pandas as pd

def generate_fake_data(num_users=100, seed=42):
    random.seed(seed)  # Ensure reproducibility
    
    # Define domains with possible sequences and weighted probabilities
    domains = ['google.com', 'youtube.com', 'facebook.com', 'instagram.com', 'twitter.com',
               'wikipedia.org', 'amazon.com', 'reddit.com', 'netflix.com', 'stackoverflow.com']
    
    # Assign probabilities to domains based on usage patterns
    domain_weights = {
        'google.com': 0.4,  # More frequent domain
        'youtube.com': 0.2,
        'facebook.com': 0.15,
        'instagram.com': 0.1,
        'wikipedia.org': 0.05,
        'amazon.com': 0.05,
        'reddit.com': 0.03,
        'netflix.com': 0.02,
        'twitter.com': 0.01,
        'stackoverflow.com': 0.01
    }

    # Define sequential patterns (e.g., if you visit google, you're likely to visit youtube or amazon next)
    sequential_patterns = {
        'google.com': ['youtube.com', 'amazon.com', 'wikipedia.org'],
        'youtube.com': ['facebook.com', 'reddit.com'],
        'facebook.com': ['instagram.com', 'twitter.com'],
        'instagram.com': ['facebook.com'],
        'wikipedia.org': ['reddit.com', 'google.com'],
        'amazon.com': ['reddit.com', 'youtube.com']
    }
    
    age_groups = ['18-25', '26-35', '36-50']
    genders = ['male', 'female']
    
    # Generate data
    data = []
    for i in range(num_users):
        user_data = {
            'user_id': i,
            'age_group': random.choice(age_groups),
            'gender': random.choice(genders),
            'browsing_sequence': [],
            'timestamps': []  # List to store timestamps for each domain visit
        }
        
        # Simulate session segments (phases of browsing activity)
        session_length = random.randint(5, 8)  # Average session length between 5-8 visits
        phase_sequence = ['info_browsing', 'social_interaction', 'shopping']
        
        # Simulate browsing segments: Information browsing, social interaction, shopping
        for phase in phase_sequence:
            if phase == 'info_browsing':
                phase_domains = ['google.com', 'wikipedia.org', 'stackoverflow.com']
            elif phase == 'social_interaction':
                phase_domains = ['facebook.com', 'instagram.com', 'twitter.com']
            else:  # shopping
                phase_domains = ['amazon.com', 'reddit.com']

            # Choose domains from the phase and apply sequential patterns
            phase_browsing = []
            current_domain = random.choice(phase_domains)  # Start with a random domain from the phase
            phase_browsing.append(current_domain)

            while len(phase_browsing) < session_length:
                # Check sequential patterns for the current domain
                next_domains = sequential_patterns.get(current_domain, domains)
                next_domain = random.choice(next_domains)  # Randomly choose the next domain
                phase_browsing.append(next_domain)
                current_domain = next_domain  # Update current domain to the next one

            user_data['browsing_sequence'].extend(phase_browsing)

        # Simulate time intervals between visits (time gap)
        start_time = random.randint(1609459200, 1640995200)  # Random start timestamp (2021)
        time_intervals = [start_time]
        
        for _ in range(1, len(user_data['browsing_sequence'])):
            time_gap = random.randint(60, 600)  # Random gap between 1 to 10 minutes (in seconds)
            time_intervals.append(time_intervals[-1] + time_gap)
        
        user_data['timestamps'] = time_intervals
        
        # Add user data to the list
        data.append(user_data)
    
    return pd.DataFrame(data)


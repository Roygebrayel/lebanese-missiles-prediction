import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load the data
data = pd.read_csv('missiles-acled-18-october-2023.csv', delimiter=';')

# **Step 1: Clean and Filter Data**
missile_data = data[(data['sub_event_type'].isin(['Shelling/artillery/missile attack', 'Air/drone strike'])) & 
                    (data['actor1'] == 'Military Forces of Israel (2022-)')]

# **Step 2: Handle Missing Values**
missile_data['latitude'] = missile_data['latitude'].fillna(missile_data['latitude'].mean())
missile_data['longitude'] = missile_data['longitude'].fillna(missile_data['longitude'].mean())
missile_data['fatalities'] = missile_data['fatalities'].fillna(0)

# **Step 3: Aggregate Data by Location**
agg_data = missile_data.groupby('location').agg({
    'event_id_cnty': 'count',  # Count of missile events per location
    'fatalities': 'sum',       # Sum of fatalities per location
    'latitude': 'mean',        # Average latitude
    'longitude': 'mean',       # Average longitude
}).reset_index()

agg_data.rename(columns={'event_id_cnty': 'missile_attack_count', 'fatalities': 'total_fatalities'}, inplace=True)

# **Step 4: Create a PTSD Likelihood Proxy**
agg_data['ptsd_likelihood'] = agg_data.apply(
    lambda row: 1 if row['missile_attack_count'] > 3 and row['total_fatalities'] > 1 else 0, axis=1
)

# **Step 5: Feature Engineering**
features = ['missile_attack_count', 'total_fatalities']

# Add interaction and log transformation features
agg_data['attack_fatality_interaction'] = agg_data['missile_attack_count'] * agg_data['total_fatalities']
agg_data['log_missile_attack_count'] = np.log1p(agg_data['missile_attack_count'])
agg_data['log_total_fatalities'] = np.log1p(agg_data['total_fatalities'])

features.extend(['attack_fatality_interaction', 'log_missile_attack_count', 'log_total_fatalities'])

# Prepare feature matrix (X) and target vector (y)
X = agg_data[features]
y = agg_data['ptsd_likelihood']

# **Step 6: Normalize/Scale Features**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# **Step 7: Train-Test Split**
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# **Step 8: Balance Dataset using SMOTE (if necessary)**
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Output the processed data to check
print(agg_data.head())

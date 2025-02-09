import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('../data/raw/clean_data_after_eda.csv', parse_dates=['date_activ', 'date_end', 'date_modif_prod', 'date_renewal'])

# Drop unnecessary columns
columns_to_drop = ['id']  # Add other columns if needed
data = data.drop(columns=columns_to_drop)

# Extract year and month from dates
data['year_activ'] = data['date_activ'].dt.year
data['month_activ'] = data['date_activ'].dt.month
data['year_end'] = data['date_end'].dt.year
data['month_end'] = data['date_end'].dt.month
data['tenure_days'] = (data['date_end'] - data['date_activ']).dt.days

# Compute off-peak price difference (December - January of preceding year)
data['price_diff_dec_jan'] = data['var_6m_price_off_peak'].shift(1) - data['var_6m_price_off_peak'].shift(12)

# Fill missing values (if any)
data.fillna(0, inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, columns=['channel_sales'], drop_first=True)

# Save processed dataset
data.to_csv('../data/processed/feature_engineered_data.csv', index=False)

print("Feature engineering complete. Dataset saved!")
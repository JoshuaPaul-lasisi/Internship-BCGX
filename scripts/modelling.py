import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('../data/raw/clean_data_after_eda.csv', parse_dates=['date_activ', 'date_end', 'date_modif_prod', 'date_renewal'])

# Drop unnecessary columns
columns_to_drop = ['id']  # Remove ID column
data = data.drop(columns=columns_to_drop)

# Extract year and month from dates
data['year_activ'] = data['date_activ'].dt.year
data['month_activ'] = data['date_activ'].dt.month
data['year_end'] = data['date_end'].dt.year
data['month_end'] = data['date_end'].dt.month
data['tenure_days'] = (data['date_end'] - data['date_activ']).dt.days

# Compute off-peak price difference (December - January of preceding year)
data['price_diff_dec_jan'] = data['var_6m_price_off_peak'].shift(1) - data['var_6m_price_off_peak'].shift(12)

# Fill missing values
data.fillna(0, inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, columns=['channel_sales'], drop_first=True)

# Define features and target variable
X = data.drop(columns=['churn'])
y = data['churn']

# Normalize numerical features (important for tree-based models too)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
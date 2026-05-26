import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("earthquake_dataset.csv")

# Label Encoding
le_region = LabelEncoder()
le_soil = LabelEncoder()
le_risk = LabelEncoder()

data['Region'] = le_region.fit_transform(data['Region'])
data['Soil_Type'] = le_soil.fit_transform(data['Soil_Type'])
data['Risk'] = le_risk.fit_transform(data['Risk'])

# Features and Target
X = data[['Latitude', 'Longitude', 'Depth',
          'Region', 'Previous_Tremors', 'Soil_Type']]

y = data['Risk']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'earthquake_model.pkl')
joblib.dump(le_region, 'region_encoder.pkl')
joblib.dump(le_soil, 'soil_encoder.pkl')
joblib.dump(le_risk, 'risk_encoder.pkl')

print("Model Trained Successfully")
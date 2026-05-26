import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("earthquake_dataset.csv")

# Label encoders
region_encoder = LabelEncoder()
soil_encoder = LabelEncoder()
risk_encoder = LabelEncoder()

# Convert text into numbers
data['Region'] = region_encoder.fit_transform(data['Region'])
data['Soil_Type'] = soil_encoder.fit_transform(data['Soil_Type'])
data['Risk'] = risk_encoder.fit_transform(data['Risk'])

# Features and target
X = data[['Latitude','Longitude','Depth',
          'Region','Previous_Tremors','Soil_Type']]

y = data['Risk']

# Train model
model = RandomForestClassifier()
model.fit(X,y)

# Save files
joblib.dump(model,'earthquake_model.pkl')
joblib.dump(region_encoder,'region_encoder.pkl')
joblib.dump(soil_encoder,'soil_encoder.pkl')
joblib.dump(risk_encoder,'risk_encoder.pkl')

print("Model Trained Successfully")
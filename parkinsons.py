#load data into dataframe
import pandas as pd
parkinsons_data = pd.read_csv('F:/project_3/parkinsons - parkinsons.csv')

# Check dataset shape
print(f"Dataset Shape: {parkinsons_data.shape}")  

# Get basic information about columns
parkinsons_data.info()

# Check for missing values
print(parkinsons_data.isnull().sum())

# Summary statistics
parkinsons_data.describe()

# Check for duplicate records
duplicates = parkinsons_data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Remove duplicates if necessary
kidney_data = parkinsons_data.drop_duplicates()

import seaborn as sns
import matplotlib.pyplot as plt
# Check distribution of target variable (Assuming 'target' is the disease label)
sns.countplot(x='status', data=parkinsons_data)
plt.title("Target Variable Distribution")
plt.show()

#Distribution of numerical features
parkinsons_data.hist(figsize=(12, 8), bins=20)
plt.show()

#Check for outliers
for col in parkinsons_data.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=kidney_data[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

#Encoding
from sklearn.preprocessing import LabelEncoder
parkinsons_data_encoded = parkinsons_data.copy()
label_encoders = {}

for col in parkinsons_data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    parkinsons_data_encoded[col] = le.fit_transform(parkinsons_data[col])
    label_encoders[col] = le  # Store encoders if needed for future decoding


#Correlation heatmap
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.heatmap(parkinsons_data_encoded.corr(), annot=True, cmap="coolwarm", fmt=".1f")
plt.title("Feature Correlation Heatmap")
plt.show()


#Feature scaling using StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Select features and target
parkinsons_features = parkinsons_data_encoded.drop(['name', 'status'], axis=1)
parkinsons_target = parkinsons_data_encoded['status']


# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit and transform the scaler on the training data

scaler.fit(parkinsons_features) 

# Transform the test data using the same scaler
scaled_features = scaler.transform(parkinsons_features)


# Split data into features and target 
from sklearn.model_selection import train_test_split
X = parkinsons_data_encoded.drop(['name', 'status'], axis=1)
y = parkinsons_data_encoded['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #Ensures balanced class distribution in training and testing

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')








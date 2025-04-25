import pandas as pd
#load data into dataframe
liver_data = pd.read_csv('F:/project_3/indian_liver_patient - indian_liver_patient.csv')

# Check dataset shape
print(f"Dataset Shape: {liver_data.shape}")  

# Get basic information about columns
liver_data.info()

# Check for missing values
print(liver_data.isnull().sum())

# Summary statistics
liver_data.describe()

# Calculate the mean of Albumin_and_Globulin_Ratio for each Dataset category
mean_values = liver_data.groupby('Dataset')['Albumin_and_Globulin_Ratio'].mean()

# Fill missing values in Albumin_and_Globulin_Ratio based on the Dataset category
liver_data['Albumin_and_Globulin_Ratio'] = liver_data.apply(
    lambda row: mean_values[row['Dataset']] if pd.isnull(row['Albumin_and_Globulin_Ratio']) else row['Albumin_and_Globulin_Ratio'],
    axis=1
)

# Check for duplicate records
duplicates = liver_data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Remove duplicates if necessary
df = liver_data.drop_duplicates()

#Check for outliers
import seaborn as sns
import matplotlib.pyplot as plt

for col in liver_data.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=liver_data[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Check distribution of target variable (Assuming 'target' is the disease label)
sns.countplot(x='Dataset', data=liver_data)
plt.title("Target Variable Distribution")
plt.show()


#Distribution of numerical features
liver_data.hist(figsize=(12, 8), bins=20)
plt.show()

#Check for outliers
for col in liver_data.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=liver_data[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

#Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(liver_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# Distribution of features
categorical_features = ['Age', 'Gender', 'Dataset'] 

for col in categorical_features:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[col])
    plt.title(f"Distribution of {col}")
    plt.show()

# Recheck for missing values
print(liver_data.isnull().sum())


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Encode 'Gender' in liver data
liver_data['Gender'] = LabelEncoder().fit_transform(liver_data['Gender'])

liver_data['Dataset'] = LabelEncoder().fit_transform(liver_data['Dataset'])


# Scale features for each dataset (excluding target variable)
liver_features = liver_data.drop('Dataset', axis=1)
liver_target = liver_data['Dataset']

# Create a MinMaxScaler object
scaler = MinMaxScaler()

scaler.fit(liver_features)
# Scale the features
scaled_features = scaler.transform(liver_features)


# Split data to features and target for training
from sklearn.model_selection import train_test_split

X = liver_features = liver_data.drop('Dataset', axis=1)
y = liver_data['Dataset']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #Ensures balanced class distribution in training and testing


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Train KNN Model
knn = KNeighborsClassifier(n_neighbors=5)  # You can tune 'n_neighbors'
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("KNN Model Accuracy:", round(accuracy * 100, 2), "%")


# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))






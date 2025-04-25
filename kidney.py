import pandas as pd
#Read database
kidney_data = pd.read_csv('F:/project_3/kidney_disease - kidney_disease.csv')


# Check dataset shape
print(f"Dataset Shape: {kidney_data.shape}")  

# Get basic information about columns
kidney_data.info()

# Check for missing values
print(kidney_data.isnull().sum())


import pandas as pd
import numpy as np

def fill_categorical_missing_values(df):
    # Fill missing values by splitting equally for binary categorical variables
    binary_cols = ['rbc', 'pc', 'htn', 'dm', 'cad', 'pe', 'ane']
    
    for col in binary_cols:
        if kidney_data[col].isna().sum() > 0:
            mode_value = kidney_data[col].mode()[0]  # Get the most frequent value
            unique_values = kidney_data[col].dropna().unique()  # Get unique non-null values
            
            if len(unique_values) == 2:  # Ensure it's binary
                # Get the count of NaN values
                nan_count = kidney_data[col].isna().sum()
                # Split equally
                half_nan = nan_count // 2
                remaining = nan_count - half_nan  # For odd counts
                
                # Replace NaNs in the column by alternating
                kidney_data.loc[kidney_data[col].isna(), col] = [unique_values[0]] * half_nan + [unique_values[1]] * remaining
            else:
                # If only one unique value is present, fill all NaNs with it
                kidney_data[col].fillna(mode_value, inplace=True)
    
    # Fill 'pcc', 'ba' (Present/NotPresent) and 'appet' (Good/Poor) using mode
    for col in ['pcc', 'ba', 'appet']:
        kidney_data[col].fillna(kidney_data[col].mode()[0], inplace=True)
    
    return kidney_data

# Apply the function to your dataset
kidney_data = fill_categorical_missing_values(kidney_data)

# Check missing values after filling
print(kidney_data.isna().sum())


#Correcting wong data

import pandas as pd
import numpy as np

def clean_numerical_values(df):
    # Correct specific incorrect values
    df.loc[df['bgr'] == 22, 'bgr'] = 398  # Blood Glucose Random
    df.loc[df['bu'] == 1.5, 'bu'] = 15  # Blood Urea
    df.loc[df['pot'] == 39, 'pot'] = 3.9  # Potassium
    df.loc[df['pot'] == 47, 'pot'] = 4.7  # Potassium
    
    # Capping outliers
    df.loc[df['sc'] > 16, 'sc'] = 16.5  # Serum Creatinine cap at 16.5
    
    # Remove invalid values
    df = df[df['sod'] != 4.5]  # Remove row with incorrect Sodium value
    df = df[df['pcv'] != '?']  # Remove invalid value in Packed Cell Volume
    df = df[df['wc'] != '?']  # Remove invalid value in White Blood Cell count
    df = df[df['rc'] != '?']  # Remove invalid value in Red Blood Cell count

    return df

# Apply the function
kidney_data = clean_numerical_values(kidney_data)

# Check if the changes were applied correctly
print(kidney_data[['bgr', 'bu', 'sc', 'pot', 'sod', 'pcv', 'wc', 'rc']].head())


#Handling missing value of numarical columns
def fill_nan_values(df):
    # Convert numeric columns to float (handling non-numeric values)
    numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, force invalid values to NaN

    # Handling 'sc' column - Mean per classification & age group
    df['age_group_sc'] = pd.cut(df['age'], bins=[0, 13, 75, float('inf')], labels=['0-13', '14-75', '76+'])
    df['sc'] = df.groupby(['classification', 'age_group_sc'])['sc'].transform(lambda x: x.fillna(x.mean()))

    # Handling 'dm' column - Split NaN values equally into 'yes' and 'no'
    nan_count = df['dm'].isna().sum()
    df.loc[df['dm'].isna(), 'dm'] = ['yes'] * (nan_count // 2) + ['no'] * (nan_count - (nan_count // 2))

    # Handling 'htn', 'bp', 'sg', 'bgr', 'su', 'al' - Using appropriate measures per age group
    df['age_group_other'] = pd.cut(df['age'], bins=[0, 18, 75, float('inf')], labels=['<18', '18-75', '>75'])

    df['htn'] = df.groupby(['classification', 'age_group_other'])['htn'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'yes'))
    df['bp'] = df.groupby(['classification', 'age_group_other'])['bp'].transform(lambda x: x.fillna(round(x.mean())))
    df['sg'] = df.groupby(['classification', 'age_group_other'])['sg'].transform(lambda x: x.fillna(x.mean()))
    df['bgr'] = df.groupby(['classification', 'age_group_other'])['bgr'].transform(lambda x: x.fillna(round(x.mean())))
    df['su'] = df.groupby(['classification', 'age_group_other'])['su'].transform(lambda x: x.fillna(x.median()))
    df['al'] = df.groupby(['classification', 'age_group_other'])['al'].transform(lambda x: x.fillna(round(x.mean())))

    # Handling 'hemo', 'pcv', 'pot', 'sod', 'wc', 'bu' - Mean per age group
    df['age_group_blood'] = pd.cut(df['age'], bins=[0, 1, 12, float('inf')], labels=['<1', '1-12', '>12'])

    for col in ['hemo', 'pcv', 'pot', 'sod', 'wc', 'bu']:
        df[col] = df.groupby(['classification', 'age_group_blood'])[col].transform(lambda x: x.fillna(round(x.mean())))

    # Handling 'rbc' - 50% Normal, 50% Abnormal
    df['age_group_rbc'] = pd.cut(df['age'], bins=[0, 14, float('inf')], labels=['<14', '≥14'])

    nan_rbc_count = df['rbc'].isna().sum()
    df.loc[df['rbc'].isna(), 'rbc'] = ['normal'] * (nan_rbc_count // 2) + ['abnormal'] * (nan_rbc_count - (nan_rbc_count // 2))

    # Handling 'rc' - Mean per age group
    df['age_group_rc'] = pd.cut(df['age'], bins=[0, 1, 12, float('inf')], labels=['<1', '1-12', '>12'])
    df['rc'] = df.groupby(['classification', 'age_group_rc'])['rc'].transform(lambda x: x.fillna(round(x.mean())))

    # **Final Check: Filling Remaining Missing Values Globally**
    for col in numeric_cols:
        df[col] = df[col].fillna(round(df[col].mean()))  # Fallback to global mean if any NaN remains

    return df

# Apply the function
kidney_data = fill_nan_values(kidney_data)

# Checking if missing values are filled
print(kidney_data.isna().sum())


# Summary statistics
kidney_data.describe()


# Check for duplicate records
duplicates = kidney_data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Remove duplicates if necessary
kidney_data = kidney_data.drop_duplicates()


import seaborn as sns
import matplotlib.pyplot as plt

# Check distribution of target variable (Assuming 'target' is the disease label)
sns.countplot(x='classification', data=kidney_data)
plt.title("Target Variable Distribution")
plt.show()


#Distribution of numerical features
kidney_data.hist(figsize=(12, 8), bins=20)
plt.show()


#Check for outliers
for col in kidney_data.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=kidney_data[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


# Encoding
from sklearn.preprocessing import LabelEncoder

kidney_data_encoded = kidney_data.copy()
label_encoders = {}

for col in kidney_data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    kidney_data_encoded[col] = le.fit_transform(kidney_data[col])
    label_encoders[col] = le  # Store encoders if needed for future decoding

# To drop unwanted column
kidney_data_encoded = kidney_data_encoded.drop(['age_group_blood', 'age_group_rbc', 'age_group_rc', 'age_group_sc', 'age_group_other'], axis=1)


#Check for outliers
for col in kidney_data_encoded.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=kidney_data_encoded[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# correlation through heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(kidney_data_encoded.corr(), annot=True, cmap="coolwarm", fmt=".1f")
plt.title("Feature Correlation Heatmap")
plt.show()


#Distribution of features through countplot
categorical_features = ['rbc', 'pc', 'pcc', 'appet', 'ba', 'htn', 'dm', 'ane', 'classification'] 

for col in categorical_features:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=kidney_data_encoded[col])
    plt.title(f"Distribution of {col}")
    plt.show()


#Feature scaling using StandardScaler
from sklearn.preprocessing import StandardScaler

# Select features and target
kidney_features = kidney_data_encoded[['rc','hemo','pcv','sg','al','htn','dm','bu','sc']] 
kidney_target = kidney_data_encoded['classification']

# Create and fit the scaler
scaler = StandardScaler()
scaler.fit(kidney_features) 

# Scale the features
scaled_features = scaler.transform(kidney_features)


# spliting data for feature and target
from sklearn.model_selection import train_test_split

X = kidney_data_encoded[['rc','hemo','pcv','sg','al','htn','dm','bu','sc']]
y = kidney_data_encoded['classification']

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

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))




# ditermining kidney disease useing sc value

def classify_sc_level(sc_value):
    if sc_value < 0.5:
        return "Low (Muscle Loss/Liver Issue)"
    elif 0.5 <= sc_value <= 1.3:
        return "Normal"
    elif 1.4 <= sc_value <= 2.0:
        return "Mild Kidney Dysfunction"
    elif 2.1 <= sc_value <= 5.0:
        return "Moderate Kidney Disease"
    elif 5.1 <= sc_value <= 7.9:
        return "Severe Kidney Disease"
    else:
        return "End-Stage Kidney Disease (Dialysis)"

kidney_data["SC_Category"] = kidney_data["sc"].apply(classify_sc_level)

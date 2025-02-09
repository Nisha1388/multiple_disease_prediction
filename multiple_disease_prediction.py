import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
@st.cache_data
def load_data():
    kidney_data = pd.read_csv("scaled_kidney_data.csv")
    liver_data = pd.read_csv("scaled_liver_data.csv")
    parkinson_data = pd.read_csv("scaled_parkinsons_data.csv")
    return kidney_data, liver_data, parkinson_data

kidney_data, liver_data, parkinson_data = load_data()

def eda_page(dataset_name, data):
    st.subheader(f"Exploratory Data Analysis - {dataset_name}")
    st.write("### Data Overview")
    st.write(data.head())

    st.write("### Missing Values")
    st.write(data.isnull().sum())

    st.write("### Statistical Summary")
    st.write(data.describe())

    st.write("### Feature Distribution")
    numeric_cols = data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, bins=30, ax=ax)
        st.pyplot(fig)

# Model Training
@st.cache_resource
def train_model(data, target, model_type):
    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model based on dataset type
    if model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)  # Using 5 neighbors
    elif model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, classification_report(y_test, y_pred, output_dict=True)

def prediction_page(model, dataset_name):
    st.subheader(f"Predict Disease - {dataset_name}")
    input_data = {}

    for col in model.feature_names_in_:
        input_data[col] = st.number_input(f"Enter {col}")

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"Prediction: {prediction}")

# Streamlit UI
st.title("Disease Prediction System")
selected_dataset = st.sidebar.selectbox("Select Dataset", ["Kidney", "Liver", "Parkinson's"])

if selected_dataset == "Kidney":
    eda_page("Kidney Disease", kidney_data)
    model, acc, report = train_model(kidney_data, 'classification', "KNN")
    prediction_page(model, "Kidney Disease")

elif selected_dataset == "Liver":
    eda_page("Liver Disease", liver_data)
    model, acc, report = train_model(liver_data, 'target', "KNN")
    prediction_page(model, "Liver Disease")

elif selected_dataset == "Parkinson's":
    eda_page("Parkinson's Disease", parkinson_data)
    model, acc, report = train_model(parkinson_data, 'status', "LogisticRegression")
    prediction_page(model, "Parkinson's Disease")

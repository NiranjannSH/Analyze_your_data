import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Streamlit app
st.title("Classification Analysis App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Function to read the dataset in chunks and process
def load_large_csv(file, chunk_size=10000):
    chunks = pd.read_csv(file, chunksize=chunk_size)
    data = pd.concat(chunks, ignore_index=True)
    return data

def handle_noisy_data(data):
    # 1. Strip whitespaces from column names and values
    data.columns = data.columns.str.strip()
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # 2. Handle missing values (impute with median for numeric columns and mode for categorical columns)
    for column in data.columns:
        if data[column].dtype == 'object':  # Categorical column
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:  # Numeric column
            data[column].fillna(data[column].median(), inplace=True)
    
    # 3. Convert any remaining non-numeric values to numeric (coerce errors)
    data = data.apply(pd.to_numeric, errors='coerce')
    
    # 4. Drop rows with any remaining NaN values (in case any column could not be converted)
    data = data.dropna()
    
    return data

if uploaded_file is not None:
    # Read and process the large CSV file
    data = load_large_csv(uploaded_file)

    # Handle noisy data
    data = handle_noisy_data(data)

    # Display the cleaned data
    st.subheader("Cleaned CSV Data")
    st.write(data)

    # Display available columns
    st.subheader("Data Columns")
    st.write("Available columns:", data.columns)

    # Display data description
    st.subheader("Data Description")
    st.write(data.describe())

    # Handle categorical variables using one-hot encoding
    categorical_columns = data.select_dtypes(include=['object']).columns
    if not categorical_columns.empty:
        st.warning("One-Hot Encoding applied to handle categorical variables.")
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        data_encoded = pd.DataFrame(encoder.fit_transform(data[categorical_columns]))
        data_encoded.columns = encoder.get_feature_names_out(categorical_columns)
        
        # Concatenate encoded columns back to the original data
        data = pd.concat([data, data_encoded], axis=1)
        
        # Drop the original categorical columns
        data = data.drop(categorical_columns, axis=1)

    # Request user to select the target column
    target_column = st.selectbox("Select Target Column", [""] + list(data.columns), index=0)

    # Proceed only if a valid target column is selected
    if target_column and target_column != "":
        # Split data into features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]

        # Convert all column names to strings (just in case they contain non-string types)
        X.columns = X.columns.astype(str)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Classification
        st.header("Classification Analysis")

        # Define classifiers
        classifiers = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "SVM": SVC(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Naive Bayes": GaussianNB(),
        }

        # Display accuracy for each classifier in a descending order
        st.subheader("Accuracy for Each Classifier (Descending Order)")

        # Calculate and store accuracies
        accuracies = {}
        for clf_name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies[clf_name] = accuracy

        # Sort accuracies in descending order
        sorted_accuracies = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)

        # Display top 10 accuracies
        top_10_classifiers = sorted_accuracies[:10]
        for clf_name, accuracy in top_10_classifiers:
            st.write(f"{clf_name}: {accuracy}")

        # Display confusion matrix for top 5 classifiers
        st.subheader("Confusion Matrix for Top 5 Classifiers")
        for clf_name, _ in top_10_classifiers[:5]:
            clf = classifiers[clf_name]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            confusion_mat = confusion_matrix(y_test, y_pred)
            st.write(f"Confusion Matrix for {clf_name}:")
            st.write(confusion_mat)

        # Write describe(), accuracy, and confusion matrix to a text file
        result_text = f"Data Description:\n{data.describe()}\n\n"
        result_text += "Accuracy for Each Classifier (Descending Order):\n"
        for clf_name, accuracy in top_10_classifiers:
            result_text += f"{clf_name}: {accuracy}\n"

        result_text += "\nConfusion Matrix for Top 5 Classifiers:\n"
        for clf_name, _ in top_10_classifiers[:5]:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            confusion_mat = confusion_matrix(y_test, y_pred)
            result_text += f"Confusion Matrix for {clf_name}:\n{confusion_mat}\n\n"

        # Button to download results as a text file
        if st.download_button("Download Results as Text", result_text, key="download_button"):
            st.success("Results downloaded successfully!")
    else:
        st.info("Please select a target column from the dropdown to proceed.")

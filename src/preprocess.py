
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    # Read data from the CSV file.
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Remove unnecessary columns
    data = data.drop(columns=['Address', 'CouncilArea', 'SellerG', 'Date'])
    
    # Handle missing values.
    data = data.dropna(subset=['Price'])
    data = data.fillna(data.median())

    return data

def encode_categorical(data):
    # Convert categorical variables to numerical values.
    categorical_cols = ['Suburb', 'Type', 'Method', 'Regionname']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    return data

def split_data(data, target_col='Price'):
    # Split the data into training and test sets.
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    # Normalize the data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

def preprocess_data(file_path):
    data = load_data(file_path)
    data = clean_data(data)
    data = encode_categorical(data)
    X_train, X_test, y_train, y_test = split_data(data)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

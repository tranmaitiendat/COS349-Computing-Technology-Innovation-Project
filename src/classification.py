# classification.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_data

def train_classification_model(file_path):
    X_train, X_test, y_train, y_test = preprocess_data(file_path)

    # Create a classification model.
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set.
    y_pred = model.predict(X_test)

    # Evaluate the model.
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", report)

    return model

if __name__ == "__main__":
    file_path = 'notebooks/Melbourne_housing.csv'
    train_classification_model(file_path)

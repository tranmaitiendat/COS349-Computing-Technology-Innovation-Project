# regression.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import preprocess_data

def train_regression_model(file_path):
    X_train, X_test, y_train, y_test = preprocess_data(file_path)

    # Create a regression model.
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set.
    y_pred = model.predict(X_test)

    # Evaluate the model.
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return model

if __name__ == "__main__":
    file_path = 'Melbourne_housing.csv'
    train_regression_model(file_path)

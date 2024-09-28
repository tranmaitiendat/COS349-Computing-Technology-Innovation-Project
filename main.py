# Import necessary modules
from preprocess import preprocess_data

from regression import train_regression
from classification import train_classification
from clustering import train_clustering
from evaluation import plot_regression_results

# Path to the data
DATA_PATH = "Melbourne_housing.csv"

def main():
    print("Starting the house price analysis and prediction process...")

    # 1. Data preprocessing
    print("Step 1: Data preprocessing...")
    data = preprocess_data(DATA_PATH)
    
    # 2. Train the regression model
    print("Step 2: Training the regression model...")
    regression_model, X_test_reg, y_test_reg = train_regression(data)

    # 3. Train the classification model
    print("Step 3: Training the classification model...")
    classification_model, X_test_clf, y_test_clf = train_classification(data)
    
    # 4. Train the clustering model
    print("Step 4: Training the clustering model...")
    clustering_model, X_clust = train_clustering(data)
    
    # 5. Evaluate the regression model
    print("Step 5: Evaluating the regression model...")
    evaluate(regression_model, X_test_reg, y_test_reg, model_type="regression")

    # 6. Evaluate the classification model
    print("Step 6: Evaluating the classification model...")
    evaluate(classification_model, X_test_clf, y_test_clf, model_type="classification")

    # 7. Evaluate the clustering model
    print("Step 7: Evaluating the clustering model...")
    evaluate(clustering_model, X_clust, model_type="clustering")

    print("Process completed!")

if __name__ == "__main__":
    main()

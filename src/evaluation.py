import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_regression_results(y_test, predictions):
    # Plot the regression results.
    plt.scatter(y_test, predictions)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title('Regression Results')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()

def plot_classification_results(y_test, predictions):
    # Plot the confusion matrix for classification.
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

def plot_clustering_results(X_test, labels):
    # Plot the clustering results.
    plt.scatter(X_test[:, 0], X_test[:, 1], c=labels, cmap='viridis')
    plt.title('Clustering Results')
    plt.show()
